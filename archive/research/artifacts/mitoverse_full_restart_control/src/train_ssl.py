"""Single-GPU DINO/iBOT B or LAST-inspired L smoke and training entry point."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

ROOT = Path('/opt/data/3_117/tanghr')
RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN / 'deps'))
sys.path.insert(0, str(ROOT / 'DINOv3/dinov3'))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dinov3.configs import get_default_config
from dinov3.fsdp.ac_compile_parallelize import activation_checkpoint_transformer
from dinov3.train.ssl_meta_arch import SSLMetaArch

from last_selection import install_ssl_hooks
from ssl_data import Collate, Images
from full_manifest import validate_manifest

WEIGHT = ROOT / 'DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
WEIGHT_SHA = '8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035'
STOP = False


def stopped(*_):
    global STOP
    STOP = True


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, default=str) + '\n')
    temporary.replace(path)


def clean_state(module) -> dict:
    return {key.replace('_checkpoint_wrapped_module.', ''): tensor.detach().cpu()
            for key, tensor in module.state_dict().items()}


def rng_state() -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def config(batch: int):
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(RUN / 'src/ssl_base_config.yaml'))
    cfg.student.pretrained_weights = ''
    cfg.student.resume_from_teacher_chkpt = ''
    cfg.student.n_storage_tokens = 4
    cfg.student.mask_k_bias = True
    cfg.teacher.pretrained_weights = ''
    cfg.gram.use_loss = False
    cfg.gram.ckpt = None
    cfg.train.batch_size_per_gpu = batch
    cfg.train.dataset_path = str(RUN / 'data/pretrain_all_mitoverse.json')
    cfg.train.output_dir = str(RUN / 'jobs')
    cfg.train.num_workers = 2
    cfg.train.cache_dataset = False
    cfg.train.compile = False
    cfg.train.sharded_eval_checkpoint = False
    cfg.train.OFFICIAL_EPOCH_LENGTH = 1000
    cfg.train.centering = 'sinkhorn_knopp'
    cfg.optim.epochs = 100
    cfg.optim.layerwise_decay = 0.98
    cfg.optim.patch_embed_lr_mult = 0.2
    cfg.optim.clip_grad = 30
    cfg.crops.global_crops_size = 256
    cfg.crops.local_crops_size = 112
    cfg.crops.local_crops_number = 8
    cfg.crops.gram_teacher_crops_size = None
    cfg.crops.rgb_mean = [0.485, 0.456, 0.406]
    cfg.crops.rgb_std = [0.229, 0.224, 0.225]
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['B', 'L', 'T', 'LT'], required=True)
    parser.add_argument('--batch', type=int, choices=[4, 8, 16], required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100000:
        raise SystemExit('steps outside planned budget')
    out = Path(os.environ['RUN_OUTPUT'])
    manifest = RUN / 'data/pretrain_all_mitoverse.json'
    manifest_audit = validate_manifest(manifest)
    manifest_sha = sha(manifest)
    method_files = ['train_ssl.py', 'ssl_data.py', 'full_manifest.py', 'ssl_base_config.yaml']
    if args.arm in {'L', 'LT'}:
        method_files.append('last_selection.py')
    if args.arm in {'T', 'LT'}:
        method_files.append('topology_appearance.py')
    method_hashes = {name: sha(RUN/'src'/name) for name in method_files}
    if args.arm in {'T', 'LT'}:
        method_hashes['topology_selected_locked.json'] = sha(RUN/'topology_calibration/topology_selected_locked.json')
    cfg = config(args.batch)
    OmegaConf.save(cfg, out / 'resolved_config.yaml')
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    seed = 1000 + args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    signal.signal(signal.SIGUSR1, stopped)
    signal.signal(signal.SIGTERM, stopped)

    dist_file = out / f'dist_init_{os.getpid()}'
    torch.distributed.init_process_group('nccl', init_method='file://' + str(dist_file), rank=0, world_size=1)
    resume = torch.load(args.resume, map_location='cpu', mmap=True, weights_only=False) if args.resume else None
    start = int(resume['completed_steps']) if resume else 0
    if start >= args.steps:
        raise SystemExit('Resume step is not below target')
    if resume:
        if resume['cfg'] != cfg_dict or resume['manifest_sha'] != manifest_sha or resume['arm'] != args.arm or resume['seed'] != args.seed:
            raise SystemExit('Resume identity mismatch')
        if resume.get('run_path') != str(RUN.resolve()):
            raise SystemExit('Resume belongs to another experiment directory')
        if resume.get('method_hashes') and resume['method_hashes'] != method_hashes:
            raise SystemExit('Resume method-code identity mismatch')
    net = SSLMetaArch(cfg)
    net.to_empty(device='cuda')
    if resume:
        net.student.load_state_dict(resume['student'], strict=True)
        net.model_ema.load_state_dict(resume['teacher'], strict=True)
    else:
        if sha(WEIGHT) != WEIGHT_SHA:
            raise SystemExit('Official natural ViT-L weight SHA256 mismatch')
        state = torch.load(WEIGHT, map_location='cpu', mmap=True, weights_only=True)
        net.student.backbone.load_state_dict(state, strict=True)
        net.student.dino_head.init_weights()
        net.student.ibot_head.init_weights()
        net.model_ema.load_state_dict(net.student.state_dict(), strict=True)
        atomic_json(out / 'initialization.json', {
            'source': 'official_natural_dinov3_vitl16', 'weight_path': str(WEIGHT),
            'weight_sha256': WEIGHT_SHA, 'strict_backbone_load': True,
            'missing_keys': [], 'unexpected_keys': [],
            'teacher_initialized_from_student': True,
            'resume': None, 'starting_step': 0,
            'manifest_sha256': manifest_sha, 'manifest_audit': manifest_audit,
        })
        del state
    net.dino_loss.init_weights()
    net.ibot_patch_loss.init_weights()
    if resume:
        net.dino_loss.load_state_dict(resume['dino_loss'])
        net.ibot_patch_loss.load_state_dict(resume['ibot_loss'])
    if args.arm in {'L', 'LT'}:
        install_ssl_hooks(net)
    if args.arm in {'T', 'LT'}:
        original_teacher_output = net.get_teacher_output
        def teacher_with_separate_views(_original_images, **kwargs):
            return original_teacher_output(net._separate_teacher_views, **kwargs)
        net.get_teacher_output = teacher_with_separate_views
    original_student_output = net.get_student_output
    def student_with_feature_stats(**kwargs):
        outputs = original_student_output(**kwargs)
        with torch.no_grad():
            global_out = outputs[0]
            net._feature_stat = {
                'cls_batch_std_mean': float(global_out['cls_pre_head'].float().std(dim=1).mean()),
                'patch_spatial_std_mean': float(global_out['patch_pre_head'].float().std(dim=2).mean()),
            }
        return outputs
    net.get_student_output = student_with_feature_stats
    activation_checkpoint_transformer(cfg, net.student.backbone)
    tracked = [(name, param, param.detach().clone())
               for name, param in list(net.student.named_parameters())[:4]]
    peak = 1.5e-5 * math.sqrt(args.batch / 16)
    groups = net.get_params_groups()
    for group in groups:
        group['lr'] = peak * group['lr_multiplier']
        group['weight_decay'] = 0.04 * group['wd_multiplier']
    optimizer = torch.optim.AdamW(groups, lr=peak, betas=(0.9, 0.99), foreach=True)
    if resume:
        optimizer.load_state_dict(resume['optimizer'])
    saved_rng = resume['rng'] if resume else None
    chain = resume['consumption_chain'] if resume else '00' * 32
    source_exposure = dict(resume.get('source_exposure', {})) if resume else {}
    volume_exposure = dict(resume.get('volume_exposure', {})) if resume else {}
    unique_volumes = set(resume.get('unique_volumes', [])) if resume else set()
    del resume
    gc.collect()
    net.train()
    dataset = Images(manifest, cfg, args.seed, start * args.batch, (args.steps - start) * args.batch, args.arm)
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=2, prefetch_factor=1,
                        persistent_workers=False, pin_memory=False, multiprocessing_context='spawn',
                        collate_fn=Collate(cfg, args.arm), generator=torch.Generator().manual_seed(seed))
    iterator = iter(loader)
    if saved_rng:
        restore_rng(saved_rng)
    completed = start
    t0 = time.monotonic()
    status = 'running'
    last = {}
    last_hourly = t0

    def checkpoint() -> None:
        payload = {'format_version': 2, 'arm': args.arm, 'seed': args.seed, 'batch': args.batch,
                   'completed_steps': completed, 'student': clean_state(net.student),
                   'teacher': clean_state(net.model_ema), 'dino_loss': clean_state(net.dino_loss),
                   'ibot_loss': clean_state(net.ibot_patch_loss), 'optimizer': optimizer.state_dict(),
                   'rng': rng_state(), 'consumption_chain': chain, 'next_sample': completed * args.batch,
                   'cfg': cfg_dict, 'manifest_sha': manifest_sha, 'natural_weight_sha': WEIGHT_SHA,
                   'run_path': str(RUN.resolve()), 'manifest_audit': manifest_audit,
                   'status': status, 'method_hashes': method_hashes,
                   'source_exposure': source_exposure, 'volume_exposure': volume_exposure,
                   'unique_volumes': sorted(unique_volumes)}
        target = out / f'checkpoint_step{completed}.pth'
        temporary = target.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(target)
        atomic_json(out / 'latest_checkpoint.json', {'path': str(target), 'step': completed, 'sha256': sha(target)})
        if completed in {10000, 30000, 60000, 100000}:
            ema_target = out / f'ema_step{completed}.pth'
            if ema_target.exists():
                raise RuntimeError(f'Immutable EMA export already exists: {ema_target}')
            ema_temp = ema_target.with_suffix('.tmp')
            torch.save({'format_version': 1, 'arm': args.arm, 'seed': args.seed,
                        'completed_steps': completed, 'teacher': payload['teacher'],
                        'cfg': cfg_dict, 'manifest_sha': manifest_sha,
                        'natural_weight_sha': WEIGHT_SHA, 'method_hashes': method_hashes}, ema_temp)
            ema_temp.replace(ema_target)
            atomic_json(out / f'ema_step{completed}.json', {'path': str(ema_target),
                        'sha256': sha(ema_target), 'completed_steps': completed})
        if args.steps == 100000:
            checkpoints = sorted(out.glob('checkpoint_step*.pth'),
                                 key=lambda p: int(p.stem.removeprefix('checkpoint_step')))
            keep = set(checkpoints[-2:]) | {p for p in checkpoints if
                      int(p.stem.removeprefix('checkpoint_step')) in {10000, 30000, 60000, 100000}}
            for old in checkpoints:
                if old not in keep:
                    old.unlink()

    with (out / 'steps.jsonl').open('a', encoding='utf-8') as log:
        for step in range(start, args.steps):
            if STOP:
                status = 'interrupted'
                break
            batch = next(iterator)
            first = batch.pop('_first_index')
            locations = batch.pop('_sample_locations')
            digest = batch.pop('_sample_digest')
            if args.arm in {'T', 'LT'}:
                separate = batch.pop('collated_teacher_global_crops')
                net._separate_teacher_views = separate.cuda(non_blocking=True).unflatten(0, (2, args.batch))
            if first != step * args.batch:
                raise RuntimeError('Sample sequence mismatch')
            chain = hashlib.sha256(bytes.fromhex(chain) + bytes.fromhex(digest)).hexdigest()
            for location in locations:
                source = location['source']
                source_exposure[source] = source_exposure.get(source, 0) + 1
                vid = location['volume_id']
                volume_exposure[vid] = volume_exposure.get(vid, 0) + 1
                unique_volumes.add(vid)
            batch['global_batch_size'] = args.batch
            lr = peak * (0.1 + 0.9 * min(step, 5000) / 5000)
            teacher_temp = 0.04 + 0.03 * min(step, 5000) / 5000
            for group in optimizer.param_groups:
                group['lr'] = lr * group['lr_multiplier']
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, metrics = net.forward_backward(batch, teacher_temp=teacher_temp, iteration=step)
            if args.arm in {'T', 'LT'}:
                net._separate_teacher_views = None
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite SSL loss')
            grad = torch.nn.utils.clip_grad_norm_(net.student.parameters(), 30, error_if_nonfinite=True)
            optimizer.step()
            net.update_ema(0.999)
            completed = step + 1
            last = {'step': completed, 'loss': float(loss.detach()), 'grad_norm': float(grad),
                    'lr': lr, 'teacher_temp': teacher_temp, 'batch': args.batch,
                    'sample_chain': chain, 'first_source': locations[0]['source'],
                    'seconds': time.monotonic() - t0,
                    'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
            last['loss_components'] = {key: float(value.detach()) if torch.is_tensor(value) else float(value)
                                       for key, value in metrics.items()
                                       if key in {'dino_local_crops_loss', 'dino_global_crops_loss',
                                                  'ibot_loss', 'koleo_loss'}}
            last['tracked_parameter_delta_l2'] = {
                name: float(torch.linalg.vector_norm((param.detach() - initial).float()))
                for name, param, initial in tracked
            }
            last['feature_stat'] = net._feature_stat
            log.write(json.dumps(last) + '\n')
            log.flush()
            if completed % 10 == 0 or completed == args.steps:
                print(json.dumps(last), flush=True)
                atomic_json(out / 'progress.json', {'status': 'running', 'target_steps': args.steps,
                            'source_exposure': source_exposure, 'volume_exposure': volume_exposure,
                            'unique_volumes': len(unique_volumes), **last})
            now = time.monotonic()
            if now - last_hourly >= 3600 or completed == args.steps:
                elapsed = max(now - t0, 1e-6)
                with (out / 'hourly_summary.jsonl').open('a', encoding='utf-8') as summary:
                    summary.write(json.dumps({'completed_steps': completed, 'target_steps': args.steps,
                                  'mean_seconds_per_step_since_start': elapsed / (completed - start),
                                  'remaining_eta_seconds': (args.steps - completed) * elapsed / (completed - start),
                                  'source_exposure': source_exposure,
                                  'unique_volumes': len(unique_volumes), 'peak_gpu_bytes': last['peak_gpu_bytes']}) + '\n')
                last_hourly = now
            if completed % 1000 == 0 or completed == args.steps:
                if completed == args.steps:
                    status = 'complete'
                checkpoint()
        else:
            status = 'complete'
    if completed and not (out / f'checkpoint_step{completed}.pth').exists():
        checkpoint()
    atomic_json(out / 'result.json', {'status': status, 'completed_steps': completed,
                                      'target_steps': args.steps, 'arm': args.arm, 'batch': args.batch, **last})
    del iterator, loader
    torch.distributed.destroy_process_group()
    return 0 if status == 'complete' else 75


if __name__ == '__main__':
    raise SystemExit(main())
