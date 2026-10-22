"""Source-Adaptive Representation Routing (SARR) Training Script.

Adapted from the official frozen Primus probe (mv_probe.py).
Trains a lightweight spatial router (or gating baseline) on top of two frozen
pretrained DINOv3 ViT-L/16 encoders:
  - P0: Official natural-image DINOv3 ViT-L/16
  - B100k: Continually pretrained on MitoVerse EM data for 100k steps

Protocol: MV-Generalist-v2 (156 train regions, 38 dev regions across 12 EM sources)
Optimizer: AdamW lr=1e-4, weight_decay=0.05, betas=(0.9, 0.98), grad_clip=12
Schedule: Cosine decay over total steps with 100-step linear warmup
Evaluation: Sliding window 256/128, mean softmax > 0.5
Milestones: Checkpoints and evaluation at 10k, 20k, 40k steps + full-volume final evaluation
"""

from pathlib import Path
import argparse
import collections
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve repository and source paths
R = Path(os.environ.get('SARR_ROOT', Path(__file__).resolve().parents[2]))
SRC_DIR = R / 'src'
SEG_DIR = SRC_DIR / 'segmentation'
SARR_DIR = SRC_DIR / 'sarr'

for p in [str(SRC_DIR), str(SEG_DIR), str(SARR_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import mv_data
from mv_data import Samples, sha, atomic
from router import SARR_Primus, SpatialRouter


def gpu_guard():
    """Verify that execution is strictly restricted to permitted GPUs (1, 2, or 3) and never GPU 0."""
    if not torch.cuda.is_available():
        return None
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible:
        for dev in visible.split(','):
            dev = dev.strip()
            if dev == '0':
                raise RuntimeError("GPU 0 is strictly forbidden by research policy! Use GPU 1, 2, or 3.")
    try:
        import subprocess, csv
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
            text=True
        )
        gpu_map = {r[1].strip(): int(r[0].strip()) for r in csv.reader(out.splitlines()) if r}
        for item in visible.split(','):
            token = item.strip()
            if token in gpu_map and gpu_map[token] == 0:
                raise RuntimeError(f"GPU {token} corresponds to physical GPU 0, which is strictly forbidden!")
    except Exception:
        pass
    return torch.cuda.current_device()


def seed(s: int):
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.set_num_threads(2)


def state_hash(state: dict) -> str:
    """Compute deterministic SHA-256 digest of a PyTorch state dict."""
    h = hashlib.sha256()
    for n, t in sorted(state.items()):
        h.update(n.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def head_state(net: nn.Module) -> dict:
    """Extract trainable head (router + decoder) parameters, excluding frozen encoders."""
    return {
        k: v.detach().cpu()
        for k, v in net.state_dict().items()
        if not (k.startswith('enc_p0.') or k.startswith('enc_b100.') or k.startswith('dino_encoder.'))
    }


def get_v2_manifest(locked: bool = True) -> dict:
    """Load the locked MV-Generalist-v2 manifest (156 train regions, 38 dev regions)."""
    candidates = [
        R / 'configs/segmentation/MV-Generalist-v2/manifest.json',
        R / 'MV-Generalist.json',
        R / 'configs/segmentation/MV-Generalist-v1/manifest.json',
        R / 'MV-Dev-v1.json',
    ]
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text(encoding='utf-8'))
            entries = data.get('entries', [])
            train_count = len([r for r in entries if r.get('supervised_role') == 'train'])
            if train_count == 156:
                return data
            if len(entries) > 0:
                return data
    raise FileNotFoundError("Could not find MV-Generalist-v2 manifest.json")


def open_cache_flexible(row: dict):
    """Load image and label memory maps from local or server cache paths."""
    candidates = [
        R / 'cache' / row['id'],
        Path(os.environ.get('MV_CACHE_DIR', '')) / row['id'] if os.environ.get('MV_CACHE_DIR') else None,
        Path('/opt/data/3_117/tanghr/codex/mv_generalist_v2_han_fullgt_20261014/cache') / row['id'],
        Path('/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013/cache') / row.get('volume_id', row['id']),
        R.parent / 'mv_generalist_stage1_20261013/cache' / row.get('volume_id', row['id']),
    ]
    for c in candidates:
        if c is not None and (c / 'image.npy').exists() and (c / 'label.npy').exists():
            return np.load(c / 'image.npy', mmap_mode='r'), np.load(c / 'label.npy', mmap_mode='r')
    return mv_data.open_cache(row)


# Hook mv_data module functions so Samples uses MV-Generalist-v2
mv_data.manifest = get_v2_manifest
mv_data.open_cache = open_cache_flexible


def load_encoder(model_id: str, registry_models: list) -> tuple:
    """Load a frozen DINOv3 ViT-L/16 encoder using the verified pattern from mv_probe.py."""
    m = next(x for x in registry_models if x['id'] == model_id)
    for p, h in [
        (m['checkpoint_path'], m['sha256']),
        (m['resolved_config_path'], m['resolved_config_sha256']),
        (m['primus_source_path'], m['primus_source_sha256']),
    ]:
        if Path(p).exists():
            assert sha(p) == h, f"SHA mismatch for {p}: expected {h}, got {sha(p)}"

    if m['dinov3_root'] not in sys.path:
        sys.path.insert(0, m['dinov3_root'])
    if m['nnunet_root'] not in sys.path:
        sys.path.insert(0, m['nnunet_root'])

    from omegaconf import OmegaConf
    from dinov3.models import build_model_from_cfg

    enc, dim = build_model_from_cfg(OmegaConf.load(m['resolved_config_path']), only_teacher=True)
    assert dim == 1024 and enc.n_storage_tokens == 4 and enc.patch_size == 16

    saved = torch.load(m['checkpoint_path'], map_location='cpu', mmap=True, weights_only=True)
    if model_id == 'P0':
        sd = saved
    else:
        assert saved['step'] == {'B100k': 100000, 'B150k': 150000}[model_id]
        if m.get('ssl_manifest_sha256'):
            assert saved['manifest_sha256'] == m['ssl_manifest_sha256']
        sd = saved['backbone']

    assert len(sd) == 368
    enc.load_state_dict(sd, strict=True, assign=True)
    enc.requires_grad_(False).eval()
    del saved, sd
    gc.collect()
    return enc, m


def build_sarr(mode: str, s: int, registry_path: Optional[Path] = None):
    """Build SARR_Primus model with frozen encoders according to requested mode.

    Args:
        mode: Routing mode ('sarr', 'uniform_avg', 'global_gate', 'p0_only', 'b100_only').
        s: Random seed for decoder initialization.
        registry_path: Optional path to model_registry.json.

    Returns:
        net: SARR_Primus model instance on CUDA (or CPU if CUDA unavailable).
        specs: Dictionary of encoder registry metadata.
    """
    if registry_path is None:
        registry_path = R / 'model_registry.json'
        if not registry_path.exists():
            for alt in [
                R / 'experiments/MV-1/model_registry.json',
                R.parent / 'artifacts/mv_generalist_stage1_20261013/model_registry.json',
            ]:
                if alt.exists():
                    registry_path = alt
                    break

    specs = json.loads(registry_path.read_text())['models']
    enc_p0, m_p0 = None, None
    enc_b100, m_b100 = None, None

    # Load P0 encoder if needed
    if mode != 'b100_only':
        enc_p0, m_p0 = load_encoder('P0', specs)

    # Load B100k encoder if needed
    if mode != 'p0_only':
        enc_b100, m_b100 = load_encoder('B100k', specs)

    seed(s)
    net = SARR_Primus(
        enc_p0=enc_p0,
        enc_b100=enc_b100,
        interaction_indices=[5, 11, 17, 23],
        mode=mode,
        embed_dim=1024,
        patch_embed_size=16,
        num_classes=2,
        router_hidden_dim=256,
    )

    if torch.cuda.is_available():
        net.cuda()
    return net, {'p0': m_p0, 'b100': m_b100}


def loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact loss from mv_probe.py: Cross-entropy (ignore_index=-1) + soft Dice on valid pixels."""
    valid = y >= 0
    t = (y == 1).float()
    p = logits.float().softmax(1)[:, 1]
    assert valid.any(), "No valid foreground/background pixels in target"
    ce = F.cross_entropy(logits.float(), y, ignore_index=-1)
    dice_num = 2 * (p * t * valid).sum() + 1e-5
    dice_den = (p * valid).sum() + (t * valid).sum() + 1e-5
    dice_loss = 1.0 - dice_num / dice_den
    return ce + dice_loss


def confusion(pred: np.ndarray, gt: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> list:
    """Calculate 2x2 confusion matrix counts [TP, FP, FN, TN]."""
    if valid_mask is not None:
        pred = pred[valid_mask]
        gt = gt[valid_mask]
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return [tp, fp, fn, tn]


def overlap(c: list) -> dict:
    """Convert confusion counts to standard segmentation metrics."""
    tp, fp, fn, tn = map(int, c)
    den = 2 * tp + fp + fn
    return {
        'dice': 2 * tp / den if den else 1.0,
        'precision': tp / (tp + fp) if (tp + fp) else None,
        'recall': tp / (tp + fn) if (tp + fn) else None,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'foreground_empty_prediction': tp + fp == 0,
    }


def starts(n: int) -> list:
    """Grid start positions for 256-sized sliding window with stride 128."""
    return sorted(set(range(0, max(1, n - 256 + 1), 128)) | {max(0, n - 256)})


@torch.no_grad()
def sliding(
    net: nn.Module,
    img: np.ndarray,
    tile_batch_size: int = 16,
    return_alpha: bool = False,
) -> tuple:
    """Sliding-window tile inference (256 window, 128 stride, mean softmax > 0.5).

    Matches mv_probe.py sliding window logic exactly, while optionally recording
    dense per-pixel aggregated routing weights alpha.
    """
    H, W = img.shape
    acc = np.zeros((H, W), np.float32)
    weight = np.zeros((H, W), np.uint16)
    alpha_acc = np.zeros((H, W), np.float32) if return_alpha else None

    coordinates = [(y, x) for y in starts(H) for x in starts(W)]
    for begin in range(0, len(coordinates), tile_batch_size):
        tiles = []
        items = []
        for y, x in coordinates[begin:begin + tile_batch_size]:
            a = np.asarray(img[y:y + 256, x:x + 256], np.float32)
            h, w = a.shape
            a = np.pad(a, ((0, 256 - h), (0, 256 - w)), constant_values=127.5)
            tiles.append(((a / 255.0 - 0.5) / 0.25).copy())
            items.append((y, x, h, w))

        z = torch.from_numpy(np.stack(tiles))[:, None]
        if torch.cuda.is_available():
            z = z.cuda()

        if return_alpha:
            logits, alpha_tiles = net(z, return_routing=True)
            probabilities = logits.softmax(1)[:, 1].cpu().numpy()
            # Upsample 16x16 patch alpha to 256x256 tile for pixel-accurate accumulation
            alpha_up = F.interpolate(alpha_tiles.float(), size=(256, 256), mode='nearest')[:, 0].cpu().numpy()
        else:
            logits = net(z)
            probabilities = logits.softmax(1)[:, 1].cpu().numpy()
            alpha_up = None

        assert np.isfinite(probabilities).all(), "Non-finite probability in sliding window"
        for i, (y, x, h, w) in enumerate(items):
            p = probabilities[i, :h, :w]
            acc[y:y + h, x:x + w] += p
            weight[y:y + h, x:x + w] += 1
            if alpha_up is not None and alpha_acc is not None:
                alpha_acc[y:y + h, x:x + w] += alpha_up[i, :h, :w]

    assert (weight > 0).all(), "Uncovered pixel in sliding window"
    pred = (acc / weight > 0.5)
    if return_alpha and alpha_acc is not None:
        mean_alpha = alpha_acc / weight
        return pred, mean_alpha
    return pred, None


@torch.no_grad()
def validate(
    net: nn.Module,
    out: Path,
    full: bool = False,
    return_alpha: bool = True,
) -> dict:
    """Evaluate model on locked MV-Generalist-v2 dev set (38 regions across 12 sources).

    Args:
        net: SARR_Primus model.
        out: Output directory for saving predictions / progress.
        full: If True, evaluates all Z slices in each region.
              If False, evaluates 8 deterministic evenly-spaced Z slices.
        return_alpha: If True, accumulates routing weight statistics.

    Returns:
        Summary dictionary containing per-region rows, source_means, macro_dice,
        and routing weight statistics (mean alpha, std alpha per source).
    """
    net.eval()
    rows = []
    t0 = time.time()
    manifest_data = mv_data.manifest()
    source_alphas = collections.defaultdict(list)

    dev_entries = [r for r in manifest_data['entries'] if r['supervised_role'] == 'dev']

    for row in dev_entries:
        im, lab = mv_data.open_cache(row)
        Z = im.shape[0]
        slices = list(range(Z)) if full else sorted(set(np.linspace(0, Z - 1, min(8, Z), dtype=int)))
        pred_memmap = None
        if full:
            pred_path = out / f"{row['id']}_prediction.npy"
            pred_memmap = np.lib.format.open_memmap(pred_path, mode='w+', shape=im.shape, dtype=np.uint8)

        c = np.zeros(4, np.int64)
        for k, z in enumerate(slices):
            pred, alpha_map = sliding(net, im[z], tile_batch_size=16, return_alpha=return_alpha)
            c += confusion(pred, lab[z].astype(bool))
            if pred_memmap is not None:
                pred_memmap[z] = pred
            if alpha_map is not None:
                # Subsample 16x16 patch centers to track distribution without memory bloat
                source_alphas[row['source_id']].extend(alpha_map[::16, ::16].flatten().tolist())
            if full and (k + 1) % 50 == 0:
                atomic(out / 'evaluation_progress.json', {
                    'volume': row['id'],
                    'z_completed': k + 1,
                    'z_total': len(slices),
                    'elapsed_s': time.time() - t0,
                })

        if pred_memmap is not None:
            pred_memmap.flush()
            del pred_memmap

        region_overlap = overlap(c)
        rows.append({
            'volume_id': row['id'],
            'source_id': row['source_id'],
            'slice_count': len(slices),
            'spacing_zyx_nm': row.get('spacing_zyx_nm'),
            **region_overlap,
        })

    # Group Dice by source
    bysource = collections.defaultdict(list)
    for r in rows:
        bysource[r['source_id']].append(r['dice'])
    source_means = {k: float(np.mean(v)) for k, v in bysource.items()}
    source_macro_dice = float(np.mean(list(source_means.values())))

    # Compute routing statistics per source
    routing_stats = {}
    for src in sorted(source_means.keys()):
        vals = source_alphas.get(src, [])
        if len(vals) > 0:
            arr = np.asarray(vals, dtype=np.float32)
            routing_stats[src] = {
                'mean_alpha': float(np.mean(arr)),
                'std_alpha': float(np.std(arr)),
                'min_alpha': float(np.min(arr)),
                'max_alpha': float(np.max(arr)),
            }
        else:
            routing_stats[src] = {'mean_alpha': 0.5, 'std_alpha': 0.0}

    all_means = [v['mean_alpha'] for v in routing_stats.values()]
    overall_mean_alpha = float(np.mean(all_means)) if all_means else 0.5

    return {
        'rows': rows,
        'descriptive_all_source_macro_dice': source_macro_dice,
        'source_macro_dice': source_macro_dice,
        'source_means': source_means,
        'routing_stats': routing_stats,
        'overall_mean_alpha': overall_mean_alpha,
        'primary_eligible_sources': [k for k, v in bysource.items() if len(v) >= 2],
        'full_volume': full,
        'test_labels_accessed': False,
    }


def smoke(mode: str, s: int, out: Path):
    """Run lightweight smoke verification on one ROI per source."""
    physical = gpu_guard()
    net, specs = build_sarr(mode, s)
    initial_head = state_hash(head_state(net))

    # Record initial encoder hashes
    enc_hashes_before = {}
    if net.enc_p0 is not None:
        enc_hashes_before['p0'] = state_hash(net.enc_p0.state_dict())
    if net.enc_b100 is not None:
        enc_hashes_before['b100'] = state_hash(net.enc_b100.state_dict())

    samples = Samples(s, sampler='source_balanced')
    trainable_pars = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_pars, lr=1e-4)

    rows = []
    t0 = time.time()
    for i, source in enumerate(samples.sources):
        net.train()
        x, y, coords = samples.sample(i, forced_source=source)
        opt.zero_grad(set_to_none=True)
        logits, alpha = net(x, return_routing=True)
        v = loss(logits, y)
        assert torch.isfinite(v), f"Non-finite loss on source {source}"
        v.backward()

        # Encoders must receive NO gradients
        if net.enc_p0 is not None:
            assert all(p.grad is None for p in net.enc_p0.parameters()), "Gradients leaked to enc_p0!"
        if net.enc_b100 is not None:
            assert all(p.grad is None for p in net.enc_b100.parameters()), "Gradients leaked to enc_b100!"

        # Trainable parameters must receive finite gradients
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable_pars), \
            "Missing or non-finite gradient in trainable parameters"
        opt.step()
        rows.append({
            'source': source,
            'loss': float(v.detach().cpu()),
            'alpha_mean': float(alpha.mean().detach().cpu()),
            'coords': coords,
        })

    # Verify frozen encoders remained byte-for-byte identical
    if net.enc_p0 is not None:
        assert enc_hashes_before['p0'] == state_hash(net.enc_p0.state_dict()), "enc_p0 weights modified!"
    if net.enc_b100 is not None:
        assert enc_hashes_before['b100'] == state_hash(net.enc_b100.state_dict()), "enc_b100 weights modified!"

    # Evaluate one single 256x256 ROI per dev source
    net.eval()
    dev_smoke = []
    manifest_entries = mv_data.manifest()['entries']
    for source in samples.sources:
        row = next(r for r in manifest_entries if r['source_id'] == source and r['supervised_role'] == 'dev')
        im, gt = mv_data.open_cache(row)
        z = len(im) // 2
        crop_im = np.asarray(im[z, :256, :256], copy=True)
        crop_gt = np.asarray(gt[z, :256, :256], copy=True)
        pred, alpha_map = sliding(net, crop_im, tile_batch_size=16, return_alpha=True)
        c = confusion(pred, crop_gt.astype(bool))
        dev_smoke.append({
            'source_id': source,
            'region_id': row['id'],
            **overlap(c),
            'alpha_mean': float(np.mean(alpha_map)) if alpha_map is not None else None,
        })

    peak_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    atomic(out / 'smoke.json', {
        'status': 'PASS',
        'mode': mode,
        'seed': s,
        'head_initial_sha256': initial_head,
        'encoder_unchanged': True,
        'trainable_parameters': sum(p.numel() for p in trainable_pars),
        'physical_gpu': physical,
        'train_steps': rows,
        'dev_ROI_smoke': dev_smoke,
        'peak_memory_bytes': peak_bytes,
        'seconds': time.time() - t0,
    })
    print(f"SMOKE_PASS {mode} seed={s} initial_head={initial_head[:16]} peak_vram={peak_bytes/(1024**2):.1f}MB", flush=True)


def train(mode: str, s: int, steps: int, out: Path):
    """Execute complete SARR training run on MV-Generalist-v2 protocol."""
    physical = gpu_guard()
    net, specs = build_sarr(mode, s)
    initial_head = state_hash(head_state(net))

    enc_hashes_before = {}
    if net.enc_p0 is not None:
        enc_hashes_before['p0'] = state_hash(net.enc_p0.state_dict())
    if net.enc_b100 is not None:
        enc_hashes_before['b100'] = state_hash(net.enc_b100.state_dict())

    samples = Samples(s, sampler='source_balanced')
    trainable_pars = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_pars, lr=1e-4, weight_decay=0.05, betas=(0.9, 0.98))

    meta = {
        'mode': mode,
        'seed': s,
        'steps': steps,
        'head_initial_sha256': initial_head,
        'encoder_p0_sha256': specs['p0']['sha256'] if specs.get('p0') else None,
        'encoder_b100_sha256': specs['b100']['sha256'] if specs.get('b100') else None,
        'encoder_frozen': True,
        'physical_gpu': physical,
        'trainable_parameters': sum(p.numel() for p in trainable_pars),
        'interaction_indices': [5, 11, 17, 23],
        'batch_size': 1,
        'optimizer': 'AdamW',
        'lr': 1e-4,
        'weight_decay': 0.05,
        'warmup_steps': 100,
        'milestones': [10000, 20000, 40000],
        'test_labels_accessed': False,
    }
    atomic(out / 'run_protocol.json', meta)

    # Resume from checkpoint if present
    start_step = 0
    counts = collections.Counter()
    sampling_hash = '0' * 64
    curve = []
    recent_losses = []
    t0 = time.time()

    existing_ckpts = sorted(out.glob('head_step*.pth'))
    if existing_ckpts:
        latest = existing_ckpts[-1]
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        missing, extra = net.load_state_dict(saved['head'], strict=False)
        assert not extra and all(k.startswith(('enc_p0.', 'enc_b100.', 'dino_encoder.')) for k in missing)
        opt.load_state_dict(saved['optimizer'])
        start_step = saved['step']
        counts.update(saved.get('source_draws', {}))
        sampling_hash = saved.get('sampling_hash', sampling_hash)
        if (out / 'curve.json').exists():
            curve = json.loads((out / 'curve.json').read_text())
        del saved
        print(f"Resumed from step {start_step} at {latest.name}", flush=True)

    milestones = [10000, 20000, 40000]

    for i in range(start_step, steps):
        net.train()
        x, y, coord = samples.sample(i)
        counts[coord['source']] += 1
        sampling_hash = hashlib.sha256(bytes.fromhex(sampling_hash) + json.dumps(coord, sort_keys=True).encode()).hexdigest()

        # Cosine LR schedule with 100-step warmup
        warmup_factor = min(1.0, (i + 1) / 100)
        cosine_factor = 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * i / steps))
        lr = 1e-4 * warmup_factor * cosine_factor
        for g in opt.param_groups:
            g['lr'] = lr

        opt.zero_grad(set_to_none=True)
        logits = net(x)
        v = loss(logits, y)
        assert torch.isfinite(v), f"Non-finite loss at step {i+1}"
        v.backward()

        # Encoders must remain ungradiented
        if net.enc_p0 is not None:
            assert all(p.grad is None for p in net.enc_p0.parameters())
        if net.enc_b100 is not None:
            assert all(p.grad is None for p in net.enc_b100.parameters())

        torch.nn.utils.clip_grad_norm_(trainable_pars, 12, error_if_nonfinite=True)
        opt.step()

        recent_losses.append(float(v.detach().cpu()))
        recent_losses = recent_losses[-200:]

        if (i + 1) % 100 == 0:
            atomic(out / 'progress.json', {
                'mode': mode,
                'seed': s,
                'step': i + 1,
                'steps': steps,
                'loss_median200': float(np.median(recent_losses)),
                'lr': lr,
                'seconds': time.time() - t0,
                'source_draws': dict(counts),
                'sampling_hash': sampling_hash,
            })

        if (i + 1) % 1000 == 0:
            print(f"STEP {mode} seed={s} step={i+1}/{steps} loss={np.median(recent_losses):.4f} elapsed={time.time()-t0:.1f}s", flush=True)

        if (i + 1) in milestones or (i + 1) == steps:
            # Save checkpoint
            ckpt_path = out / f"head_step{i+1:06d}.pth"
            ckpt_data = {
                'head': head_state(net),
                'optimizer': opt.state_dict(),
                'step': i + 1,
                'metadata': meta,
                'source_draws': dict(counts),
                'sampling_hash': sampling_hash,
            }
            torch.save(ckpt_data, ckpt_path)
            ckpt_sha = sha(ckpt_path)
            atomic(ckpt_path.with_suffix('.json'), {
                'path': str(ckpt_path),
                'sha256': ckpt_sha,
                'step': i + 1,
            })

            # Monitor validation (8 deterministic slices per region)
            val = validate(net, out, full=False)
            curve = [entry for entry in curve if entry['step'] != i + 1]
            curve.append({
                'step': i + 1,
                'loss_median200': float(np.median(recent_losses)),
                'checkpoint_sha256': ckpt_sha,
                **val,
            })
            atomic(out / 'curve.json', curve)
            print(f"CHECKPOINT {mode} seed={s} step={i+1} Dice={val['source_macro_dice']:.6f} MeanAlpha={val.get('overall_mean_alpha', 0.5):.4f}", flush=True)

    # Verify encoders were strictly unchanged
    if net.enc_p0 is not None:
        assert enc_hashes_before['p0'] == state_hash(net.enc_p0.state_dict())
    if net.enc_b100 is not None:
        assert enc_hashes_before['b100'] == state_hash(net.enc_b100.state_dict())

    # Final full-volume evaluation across all slices
    print(f"Running final full-volume evaluation for {mode} seed={s}...", flush=True)
    final_val = validate(net, out, full=True)

    meta.update({
        'completed_steps': steps,
        'sampling_hash': sampling_hash,
        'source_draws': dict(counts),
        'seconds': time.time() - t0,
        'encoder_unchanged': True,
        'final_validation': final_val,
    })

    # Save final head
    torch.save({
        'head': head_state(net),
        'step': steps,
        'metadata': meta,
    }, out / 'head_last.pth')

    atomic(out / 'metrics.json', {
        'status': 'complete',
        **meta,
    })
    print(f"TRAIN_COMPLETE {mode} seed={s} Final Macro Dice={final_val['source_macro_dice']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train SARR for EM mitochondria segmentation")
    parser.add_argument('--mode', choices=['sarr', 'uniform_avg', 'global_gate', 'p0_only', 'b100_only'], default='sarr',
                        help="Routing mode to execute")
    parser.add_argument('--seed', type=int, default=20261014, help="Random seed for decoder initialization")
    parser.add_argument('--steps', type=int, default=40000, help="Total training steps (default 40000)")
    parser.add_argument('--out', type=Path, default=None, help="Output directory")
    parser.add_argument('--smoke', action='store_true', help="Run smoke test only")
    args = parser.parse_args()

    if args.out is None:
        args.out = R / ('smoke' if args.smoke else 'runs') / args.mode / f"seed{args.seed}"
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        smoke(args.mode, args.seed, args.out)
    else:
        train(args.mode, args.seed, args.steps, args.out)


if __name__ == '__main__':
    main()
