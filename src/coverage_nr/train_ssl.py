"""Vanilla DINOv3 DINO+iBOT+KoLeo continual SSL on the full image slice pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dinov3.configs import get_default_config
from dinov3.fsdp.ac_compile_parallelize import activation_checkpoint_transformer
from dinov3.train.ssl_meta_arch import SSLMetaArch
from ssl_full_data import Collate, Images
from collections import Counter
from common import R, PARENT
from runtime_guard import training_gpu_guard


WEIGHT_SHA = '8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035'
STOP = False
MILESTONES = {10000, 30000, 100000}
from policy import learning_rate, validate_parent


def stop_requested(*_):
    global STOP
    STOP = True


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def full_state(module) -> dict:
    return {key.replace('_checkpoint_wrapped_module.', ''): tensor.detach().cpu()
            for key, tensor in module.state_dict().items()}


def rng_state() -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(value: dict) -> None:
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    torch.cuda.set_rng_state(value['cuda'])


def guard_is_stale(path_text: str | None) -> bool:
    if not path_text:
        return False
    try:
        heartbeat_time = float(Path(path_text).read_text().strip())
    except (OSError, ValueError):
        return True
    return time.time() - heartbeat_time > 150


def reduce_gradients(parameters, world_size: int) -> None:
    if world_size == 1:
        return
    for param in parameters:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def check_config(cfg, world_size: int, batch: int, steps: int) -> None:
    if cfg.dino.loss_weight != 1 or cfg.ibot.loss_weight != 1 or cfg.dino.koleo_loss_weight != 0.1:
        raise ValueError('Baseline losses differ from vanilla DINOv3')
    if cfg.gram.use_loss or cfg.distillation.enabled or cfg.multidistillation.enabled:
        raise ValueError('Method loss or distillation enabled in B_full')
    if cfg.student.n_storage_tokens != 4 or not cfg.student.mask_k_bias or cfg.student.in_chans != 3:
        raise ValueError('Official ViT-L/16 backbone architecture drift')
    if cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.optim.epochs != 100000:
        raise ValueError('Formal budget must resolve to exactly 100000 optimizer steps')
    if steps > 100000 or batch != cfg.train.batch_size_per_gpu or world_size not in (1, 3):
        raise ValueError('Unexpected smoke/formal batch or world size')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--weight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError('Fresh training output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    training_gpu_guard(world_size)
    assert args.output.resolve().is_relative_to(R.resolve())
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', init_method='env://' if world_size > 1 else
                            f'file://{args.output / f"single_rank_dist_{os.getpid()}"}', rank=rank,
                            world_size=world_size, device_id=torch.device('cuda', local_rank))
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGUSR1, stop_requested)
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(args.config))
    batch_size = int(cfg.train.batch_size_per_gpu)
    check_config(cfg, world_size, batch_size, args.steps)
    cfg_hash = sha256(args.config)
    manifest_hash = sha256(args.manifest)
    if sha256(args.weight) != WEIGHT_SHA:
        raise ValueError('Official natural weight hash mismatch')
    if rank == 0:
        OmegaConf.save(cfg,args.output/'resolved_config.yaml')
    manifest_doc = json.loads(args.manifest.read_text())
    if manifest_doc['training_role'] != 'ssl_diversity_d12_v1' or manifest_doc['volume_count'] != len(manifest_doc['volumes']):
        raise ValueError('Manifest is not full MitoVerse')
    seed = 1000 + args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    with torch.device('meta'):
        net = SSLMetaArch(cfg)
    net.to_empty(device='cuda')
    state = torch.load(args.weight, map_location='cpu', mmap=True, weights_only=True)
    net.student.backbone.load_state_dict(state, strict=True)
    net.student.dino_head.init_weights()
    net.student.ibot_head.init_weights()
    net.dino_loss.init_weights()
    net.ibot_patch_loss.init_weights()
    net.model_ema.load_state_dict(net.student.state_dict(), strict=True)
    del state
    activation_checkpoint_transformer(cfg, net.student.backbone)
    params = list(net.student.parameters())
    groups = net.get_params_groups()
    optimizer = torch.optim.AdamW(groups, lr=float(cfg.optim.lr),
                                  betas=(float(cfg.optim.adamw_beta1), float(cfg.optim.adamw_beta2)), foreach=True)
    identity = {'experiment_id': args.output.parent.name, 'manifest_sha256': manifest_hash,
                'config_sha256': cfg_hash, 'official_weight_sha256': WEIGHT_SHA,
                'trainer_sha256': sha256(Path(__file__)),
                'sampler_sha256': sha256(Path(__file__).with_name('ssl_full_data.py')),
                'dinov3_commit': Path(__file__).with_name('dinov3_commit.txt').read_text().strip(),
                'world_size': world_size, 'batch_per_gpu': batch_size,
                'gradient_accumulation': 1, 'effective_global_batch': batch_size * world_size,
                'base_lr_global16': 1.5e-5, 'effective_peak_lr': float(cfg.optim.lr),
                'warmup_steps': 5000, 'total_steps': 100000,
                'weight_decay': float(cfg.optim.weight_decay), 'teacher_momentum': 0.999,
                'seed': args.seed,
                'coverage_plan_sha256': manifest_doc['coverage_plan_sha256'],
                'sampler_core_sha256': sha256(Path(__file__).with_name('coverage_sampler.py')),
                'python_executable':sys.executable, 'torch_version':torch.__version__,
                'run_path': str(args.output.resolve())}
    start = 0
    prior_rng = None
    chain = '00' * 32
    if args.resume:
        receipt = json.loads((args.resume.parent / 'metadata.json').read_text())
        if receipt['sha256'] != sha256(args.resume):
            raise ValueError('Checkpoint checksum mismatch')
        saved = torch.load(args.resume, map_location='cpu', mmap=True, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Cross-run continuation identity mismatch')
        from coverage_sampler import Plan
        resume_plan=Plan(R/'plan',R/'mitoverse_pool.json',R/'cem_pool.json')
        resume_plan.validate_resume(saved['sampler_state'])
        assert saved['sampler_state']['consumed_step']==saved['step']
        start = int(saved['step'])
        if not 0 < start < args.steps:
            raise ValueError('Invalid resume step')
        net.student.load_state_dict(saved['student'], strict=True)
        net.model_ema.load_state_dict(saved['teacher'], strict=True)
        net.dino_loss.load_state_dict(saved['dino_loss'], strict=True)
        net.ibot_patch_loss.load_state_dict(saved['ibot_loss'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        prior_rng = saved['rng_by_rank'][rank]
        chain = saved['sample_chain']
        del saved
    if rank == 0:
        atomic_json(args.output / 'initialization.json', {
            **identity, 'initialization': 'full_state_resume' if args.resume else 'fresh_official_P0_student_EMA_optimizer_centers', 'strict_backbone_load': True,
            'resume': str(args.resume) if args.resume else None, 'starting_step': start,
            'dino': 1, 'ibot': 1, 'koleo': 0.1, 'gram': False, 'LAST': False,
            'topology': False, 'segmentation': False})
    dist.barrier()
    tracked_student = next(iter(net.student.backbone.parameters()))
    tracked_teacher = next(iter(net.model_ema.backbone.parameters()))
    initial_student = tracked_student.detach().clone()
    initial_teacher = tracked_teacher.detach().clone()
    net.train()
    dataset = Images(args.manifest, cfg, args.seed, start * batch_size,
                     (args.steps - start) * batch_size, rank=rank,
                     world_size=world_size, batch_per_rank=batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, prefetch_factor=1,
                        persistent_workers=False, pin_memory=False, multiprocessing_context='spawn',
                        collate_fn=Collate(cfg), generator=torch.Generator().manual_seed(seed + rank))
    iterator = iter(loader)
    if prior_rng:
        restore_rng(prior_rng)
    t0 = time.monotonic()
    total_batch = batch_size * world_size
    completed = start
    source_draws = Counter()

    def checkpoint(step: int, sample_chain: str) -> None:
        rng_all = [None] * world_size
        dist.all_gather_object(rng_all, rng_state())
        if rank == 0:
            directory = args.output / 'checkpoints' / f'step{step:09d}'
            directory.mkdir(parents=True, exist_ok=False)
            target = directory / 'full.pth'
            temporary = directory / 'full.tmp'
            torch.save({'format_version': 1, 'identity': identity, 'step': step,
                        'student': full_state(net.student), 'teacher': full_state(net.model_ema),
                        'dino_loss': full_state(net.dino_loss), 'ibot_loss': full_state(net.ibot_patch_loss),
                        'optimizer': optimizer.state_dict(), 'rng_by_rank': rng_all,
                        'sample_chain': sample_chain, 'sampler_state': dataset.plan.checkpoint_state(step)}, temporary)
            temporary.replace(target)
            atomic_json(directory / 'metadata.json', {**identity, 'step': step, 'sha256': sha256(target),
                                                        'sample_chain': sample_chain, 'kind': 'full_optimizer'})
            atomic_json(args.output / 'latest_checkpoint.json', {'path': str(target), 'step': step,
                                                                   'sha256': sha256(target)})
            if step in MILESTONES:
                for name, module in [('student_backbone', net.student.backbone),
                                     ('teacher_backbone', net.model_ema.backbone)]:
                    export = directory / f'{name}.pth'
                    torch.save({'step': step, 'backbone': full_state(module),
                                'manifest_sha256': manifest_hash,
                                'official_weight_sha256': WEIGHT_SHA,
                                'config_sha256': cfg_hash}, export)
                    atomic_json(export.with_suffix('.json'),{'path':str(export),'step':step,'sha256':sha256(export),'kind':name,'manifest_sha256':manifest_hash,'config_sha256':cfg_hash})
            # Keep two most recent full optimizer states; milestone backbone exports stay.
            all_dirs = sorted((args.output / 'checkpoints').glob('step*'))
            for old in all_dirs[:-2]:
                old_full = old / 'full.pth'
                if old_full.exists() and int(old.name[4:]) not in MILESTONES:
                    old_full.unlink()
        dist.barrier()

    with (args.output / f'steps_rank{rank}.jsonl').open('a') as log:
        for step in range(start, args.steps):
            stale_guard = guard_is_stale(os.environ.get('STAGE1_GUARD_HEARTBEAT'))
            stop_flag = torch.tensor(int(STOP or stale_guard or (args.output / 'STOP').exists()), device='cuda')
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            if int(stop_flag.item()):
                break
            iteration_started = time.monotonic()
            batch = next(iterator)
            data_seconds = time.monotonic() - iteration_started
            first = batch.pop('_first_index')
            locations = batch.pop('_sample_locations')
            digest = batch.pop('_sample_digest')
            source_draws.update(p['domain']+'::'+p['source'] for p in locations)
            expected_first = step * total_batch + rank * batch_size
            if first != expected_first:
                raise ValueError(f'Sample sequence mismatch: {first} != {expected_first}')
            digests = [None] * world_size
            dist.all_gather_object(digests, digest)
            for item in digests:
                chain = hashlib.sha256(bytes.fromhex(chain) + bytes.fromhex(item)).hexdigest()
            batch['global_batch_size'] = total_batch
            lr = learning_rate(step, float(cfg.optim.lr), float(cfg.optim.min_lr))
            for group in optimizer.param_groups:
                group['lr'] = (0 if group['is_last_layer'] and step < 1000 else lr * group['lr_multiplier'])
                group['weight_decay'] = float(cfg.optim.weight_decay) * group['wd_multiplier']
            optimizer.zero_grad(set_to_none=True)
            teacher_temp = 0.04 + 0.03 * min(step, 5000) / 5000
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, metrics = net.forward_backward(batch, teacher_temp=teacher_temp, iteration=step)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite SSL loss at step {step}')
            reduce_gradients(params, world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float(cfg.optim.clip_grad), error_if_nonfinite=True)
            optimizer.step()
            net.update_ema(0.999)
            completed = step + 1
            iteration_seconds = time.monotonic() - iteration_started
            last = {'step': completed, 'rank': rank, 'loss': float(loss.detach()),
                    'grad_norm': float(grad_norm), 'lr': lr, 'teacher_momentum': 0.999,
                    'teacher_temp': teacher_temp, 'global_batch': total_batch,
                    'data_first_index': first, 'data_first_volume': locations[0]['volume_id'],
                    'data_first_domain': locations[0]['domain'],
                    'sample_chain': chain, 'elapsed_seconds': time.monotonic() - t0,
                    'data_seconds': data_seconds, 'iteration_seconds': iteration_seconds,
                    'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                    'loss_components': {key: float(value.detach()) if torch.is_tensor(value) else float(value)
                                        for key, value in metrics.items()
                                        if key in ('dino_local_crops_loss', 'dino_global_crops_loss', 'ibot_loss', 'koleo_loss')}}
            log.write(json.dumps(last) + '\n')
            log.flush()
            if rank == 0 and (completed % 10 == 0 or completed == args.steps):
                atomic_json(args.output / 'progress.json', last)
            if completed % 1000 == 0 or completed == args.steps:
                atomic_json(args.output / f'sampling_totals_rank{rank}.json',dict(source_draws))
            if completed % 1000 == 0 or completed in MILESTONES or completed == args.steps:
                checkpoint(completed, chain)
        if completed and completed % 1000 and completed != args.steps:
            checkpoint(completed, chain)
    synced = [torch.empty_like(tracked_student.detach()) for _ in range(world_size)]
    dist.all_gather(synced, tracked_student.detach())
    max_rank_difference = max(float((synced[0] - item).abs().max()) for item in synced)
    if max_rank_difference > 1e-6:
        raise RuntimeError(f'Student parameters diverged across ranks: {max_rank_difference}')
    if rank == 0:
        atomic_json(args.output / 'result.json', {'status': 'complete' if completed == args.steps else 'interrupted',
                                                   'completed_steps': completed,
                                                   'max_rank_parameter_difference': max_rank_difference,
                                                   'tracked_student_delta_l2': float((tracked_student.detach() - initial_student).float().norm()),
                                                   'tracked_teacher_delta_l2': float((tracked_teacher.detach() - initial_teacher).float().norm()),
                                                   **identity})
    dist.barrier()
    dist.destroy_process_group()
    return 0 if completed == args.steps else 75


if __name__ == '__main__':
    raise SystemExit(main())
