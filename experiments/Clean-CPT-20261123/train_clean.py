"""Train the matched Clean-CPT arms with the historical D2 loader and Barrier."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent.parent
ORIGINAL = ROOT.parent / "ssl_diversity_d12_20261017"
CONT = ROOT.parent / "ssl_diversity_d12_100k_20261018"
BARRIER = ROOT.parent / "pilot_c100_paired_20261028" / "code"
os.environ["D12_ROOT"] = str(CONT)
sys.path.extend([str(CONT / "code"), str(BARRIER)])

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dinov3.configs import get_default_config
from dinov3.fsdp.ac_compile_parallelize import activation_checkpoint_transformer
from dinov3.models import build_model_from_cfg
from dinov3.train.ssl_meta_arch import SSLMetaArch
from legacy_ssl_data import Images as LegacyImages
from ssl_full_data import Collate, Images
from production_adapter import ProductionBarrierAdapter
from barrier import separation_barrier

spec = importlib.util.spec_from_file_location("continuation_historical", CONT / "code/train_ssl.py")
historical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(historical)

MANIFEST = ROOT / "clean_cpt_manifest.json"
CONFIG = ORIGINAL / "D12_base.yaml"
P0 = PROJECT / "DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
P0_SHA = "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
C0 = "Clean_C0_100k"
EARLY = "Clean_C1_0k_30k"
LATE = "Clean_C1_30k_to_100k"
READY = ROOT / "step000030000_full_state.READY"
BATCH = 16
WORLD = 3


def read(path):
    return json.loads(path.read_text())


def atomic(path, data):
    historical.atomic_json(path, data)


class CleanImages(Images):
    """Initialize inherited augmentations, then substitute only the allowed pools."""
    def __init__(self, cfg, start, end, rank):
        LegacyImages.__init__(self, CONT / "inputs/pretrain_all_mitoverse.json", cfg, 0,
                              start * BATCH, (end - start) * BATCH, rank=rank,
                              world_size=WORLD, batch_per_rank=BATCH)
        document = read(MANIFEST)
        self.spec = document
        self.pool = document["mitoverse"]
        self.cem = document["cem"]
        assert document["cem_fraction"] == 0.5
        self.cem["index"] = str(PROJECT / self.cem["index"])
        self.rows = self.pool["volumes"]
        for row in self.rows:
            row["local_path"] = str(PROJECT / row["local_path"])
        self.row_index = {row["volume_id"]: row for row in self.rows}
        self.archive_handle = None
        self.db = None


class WarmupBarrier(ProductionBarrierAdapter):
    weight = 0.0

    def _barrier_backward(self, gradient):
        weight = float(gradient.detach())
        assert math.isclose(weight, self.weight, abs_tol=1e-8, rel_tol=1e-7)
        self.latest["barrier_backward_seen"] = True

    def compute_losses(self, **kwargs):
        base, metrics = self.original_losses(**kwargs)
        tokens = kwargs["student_global"]["patch_pre_head"].flatten(0, 1)
        assert tokens.shape == self.anchor_tokens.shape and tokens.shape[1:] == (256, 1024)
        raw, stats = separation_barrier(tokens, self.anchor_tokens, grid_hw=(16, 16),
                                        radius=2, quantile=0.5, margin=0.05)
        assert raw.requires_grad and torch.isfinite(raw)
        raw.register_hook(self._barrier_backward)
        self.latest = {
            **stats, "barrier_raw": float(raw.detach()),
            "barrier_weighted": self.weight * float(raw.detach()),
            "lambda": self.weight, "barrier_backward_seen": False,
        }
        return base + self.weight * raw, metrics | {"barrier_raw": raw.detach()}

    def finish_step(self):
        assert self.latest["barrier_backward_seen"] and self.student_calls == 1
        assert not any(p.grad is not None for p in self.anchor.parameters())
        self.anchor_tokens = None
        return self.latest


def parent(identity):
    receipt = read(READY)
    assert receipt["step"] == 30000 and receipt["model"] == C0
    assert receipt["identity"] == identity
    folder = ROOT / "runs" / C0 / "checkpoints" / "step000030000"
    return receipt, folder


def run(arm, mode):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2,3"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    rank = int(os.environ["RANK"])
    assert int(os.environ["LOCAL_RANK"]) == rank and int(os.environ["WORLD_SIZE"]) == WORLD
    assert torch.cuda.device_count() == WORLD
    torch.cuda.set_device(rank)
    expected_uuid = os.environ["CLEAN_GPU_UUIDS"].split(",")[rank].removeprefix("GPU-")
    assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix("GPU-").lower() == expected_uuid.lower()
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    seed = 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(CONFIG))
    assert cfg.train.batch_size_per_gpu == BATCH and cfg.crops.global_crops_size == 256
    assert cfg.crops.local_crops_size == 112 and not cfg.gram.use_loss
    assert cfg.dino.loss_weight == cfg.ibot.loss_weight == 1 and cfg.dino.koleo_loss_weight == 0.1
    assert math.isclose(float(cfg.optim.lr), 2.5980762e-5, rel_tol=1e-7)
    assert float(cfg.optim.weight_decay) == 0.04
    identity = {"manifest_sha256": historical.sha256(MANIFEST), "config_sha256": historical.sha256(CONFIG),
                "trainer_sha256": historical.sha256(Path(__file__)), "p0_sha256": P0_SHA,
                "schedule": "historical_D2_30k_then_100k", "world_size": WORLD, "global_batch": 48}
    smoke = mode == "smoke"
    start = 30000 if arm == LATE else 0
    end = 200 if smoke else (30000 if arm == EARLY else 100000)
    assert smoke is False or start == 0
    out = ROOT / ("smoke" if smoke else "runs") / arm
    assert out.is_dir()
    config = {"arm": arm, "start": start, "end": end, **identity}
    if rank == 0:
        atomic(out / "config.json", config)
    with torch.device("meta"):
        net = SSLMetaArch(cfg)
    net.to_empty(device="cuda")
    natural = torch.load(P0, map_location="cpu", mmap=True, weights_only=True)
    net.student.backbone.load_state_dict(natural, strict=True)
    net.student.dino_head.init_weights()
    net.student.ibot_head.init_weights()
    net.dino_loss.init_weights()
    net.ibot_patch_loss.init_weights()
    net.model_ema.load_state_dict(net.student.state_dict(), strict=True)
    activation_checkpoint_transformer(cfg, net.student.backbone)
    params = list(net.student.parameters())
    opt = torch.optim.AdamW(net.get_params_groups(), lr=float(cfg.optim.lr),
                            betas=(float(cfg.optim.adamw_beta1), float(cfg.optim.adamw_beta2)), foreach=True)
    chain = "00" * 32
    previous_rng = None
    anchor_state = natural
    if arm == LATE:
        receipt, folder = parent(identity)
        saved = torch.load(folder / "full.pth", map_location="cpu", mmap=True, weights_only=False)
        assert saved["step"] == start and saved["identity"] == identity and saved["model"] == C0
        assert saved["scheduler"]["next_step"] == start
        net.student.load_state_dict(saved["student"], strict=True)
        net.model_ema.load_state_dict(saved["teacher"], strict=True)
        net.dino_loss.load_state_dict(saved["dino_loss"], strict=True)
        net.ibot_patch_loss.load_state_dict(saved["ibot_loss"], strict=True)
        opt.load_state_dict(saved["optimizer"])
        assert {int(v["step"]) for v in opt.state.values()} == {start}
        chain = saved["sample_chain"]
        previous_rng = saved["rng_by_rank"][rank]
        del saved
        anchor_state = torch.load(folder / "teacher_backbone.pth", map_location="cpu",
                                  mmap=True, weights_only=True)["backbone"]
        atomic(out / f"resume_rank{rank}.json", {"step": start, "parent_receipt": receipt,
                                                 "sample_chain": chain})
    adapter = None
    if arm != C0:
        with torch.device("meta"):
            anchor, dim = build_model_from_cfg(cfg, only_teacher=True)
        anchor.to_empty(device="cuda")
        anchor.load_state_dict(anchor_state, strict=True)
        anchor.requires_grad_(False).eval()
        assert dim == 1024 and anchor.patch_size == 16 and anchor.n_storage_tokens == 4
        adapter = WarmupBarrier(net, anchor) if arm == EARLY else ProductionBarrierAdapter(net, anchor)
    del natural, anchor_state
    net.train()
    dataset = CleanImages(cfg, start, end, rank)
    loader = DataLoader(dataset, batch_size=BATCH, num_workers=2, prefetch_factor=1,
                        persistent_workers=False, pin_memory=False, multiprocessing_context="spawn",
                        collate_fn=Collate(cfg), generator=torch.Generator().manual_seed(seed + rank))
    iterator = iter(loader)
    first_batch = next(iterator)
    if previous_rng is not None:
        historical.restore_rng(previous_rng)
    milestones = {15000, 30000, 45000, 60000, 80000, 100000}

    def save(step):
        states = [None] * WORLD
        if step == 30000:
            dist.all_gather_object(states, historical.rng_state())
        if rank == 0:
            folder = out / "checkpoints" / f"step{step:09d}"
            folder.mkdir(parents=True, exist_ok=False)
            teacher = folder / "teacher_backbone.pth"
            torch.save({"step": step, "backbone": historical.full_state(net.model_ema.backbone),
                        "model": arm, **identity}, teacher.with_suffix(".tmp"))
            teacher.with_suffix(".tmp").replace(teacher)
            receipt = {"step": step, "model": arm, "identity": identity, "sample_chain": chain,
                       "teacher_sha256": historical.sha256(teacher)}
            if step == 30000:
                target = folder / "full.pth"
                torch.save({"step": step, "model": arm, "identity": identity, "sample_chain": chain,
                            "student": historical.full_state(net.student), "teacher": historical.full_state(net.model_ema),
                            "dino_loss": historical.full_state(net.dino_loss),
                            "ibot_loss": historical.full_state(net.ibot_patch_loss),
                            "optimizer": opt.state_dict(), "rng_by_rank": states,
                            "scheduler": {"next_step": step, "schedule": identity["schedule"]}},
                           target.with_suffix(".tmp"))
                target.with_suffix(".tmp").replace(target)
                receipt["full_sha256"] = historical.sha256(target)
            atomic(folder / "metadata.json", receipt)
            if step == 30000 and arm == C0:
                atomic(READY, receipt)
        dist.barrier()

    peak = 0
    start_time = time.monotonic()
    with (out / f"steps_rank{rank}.jsonl").open("x", buffering=1) as stream:
        for step in range(start, end):
            began = time.monotonic()
            batch = first_batch if step == start else next(iterator)
            index = batch.pop("_first_index")
            batch.pop("_sample_locations")
            digest = batch.pop("_sample_digest")
            assert index == step * BATCH * WORLD + rank * BATCH
            digests = [None] * WORLD
            dist.all_gather_object(digests, digest)
            for value in digests:
                chain = hashlib.sha256(bytes.fromhex(chain) + bytes.fromhex(value)).hexdigest()
            batch["global_batch_size"] = BATCH * WORLD
            lr = historical.learning_rate(step, float(cfg.optim.lr), float(cfg.optim.min_lr))
            if 5000 <= step < 30000:
                lr = float(cfg.optim.min_lr) + (float(cfg.optim.lr) - float(cfg.optim.min_lr)) * 0.5 * (
                    1 + math.cos(math.pi * (step - 5000) / 25000))
            for group in opt.param_groups:
                group["lr"] = 0 if group["is_last_layer"] and step < 1000 else lr * group["lr_multiplier"]
                group["weight_decay"] = float(cfg.optim.weight_decay) * group["wd_multiplier"]
            if arm == EARLY:
                adapter.weight = min(1.0, step / 5000)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = net.forward_backward(batch, teacher_temp=0.04 + 0.03 * min(step, 5000) / 5000,
                                                     iteration=step)
            assert torch.isfinite(loss)
            barrier = adapter.finish_step() if adapter else {}
            historical.reduce_gradients(params, WORLD)
            norm = torch.nn.utils.clip_grad_norm_(params, float(cfg.optim.clip_grad), error_if_nonfinite=True)
            opt.step()
            net.update_ema(0.999)
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            reserved = torch.cuda.memory_reserved()
            peak = max(peak, total - free - reserved + torch.cuda.max_memory_reserved())
            completed = step + 1
            values = {"step": completed, "rank": rank, "sample_digest": digest, "sample_chain": chain,
                      "loss": float(loss.detach()), "grad_norm": float(norm), "lr": lr,
                      "wall_seconds": time.monotonic() - start_time, "step_seconds": time.monotonic() - began,
                      "gpu_peak_used_bytes": peak, "gpu_total_bytes": total,
                      "loss_components": {key: float(value) for key, value in metrics.items() if "loss" in key}, **barrier}
            if smoke or completed % 500 == 0:
                stream.write(json.dumps(values, allow_nan=False) + "\n")
            if completed % 100 == 0:
                atomic(out / f"progress_rank{rank}.json", values)
            if not smoke and completed in milestones:
                save(completed)
    assert {int(v["step"]) for v in opt.state.values()} == {end}
    atomic(out / f"result_rank{rank}.json", values)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=(C0, EARLY, LATE))
    parser.add_argument("mode", choices=("smoke", "formal"))
    args = parser.parse_args()
    run(args.arm, args.mode)
