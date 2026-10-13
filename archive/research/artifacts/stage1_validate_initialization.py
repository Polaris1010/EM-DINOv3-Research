"""Strictly load official ViT-L weights and compare inference/training features."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXPECTED = '8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dinov3-root', type=Path, required=True)
    parser.add_argument('--override', type=Path, required=True)
    parser.add_argument('--weight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolved', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.dinov3_root))
    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    from dinov3.configs import get_default_config
    from dinov3.models import build_model_from_cfg
    from dinov3.train.ssl_meta_arch import SSLMetaArch

    actual_hash = sha256(args.weight)
    if actual_hash != EXPECTED:
        raise ValueError(f'Official weight SHA256 mismatch: {actual_hash}')
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(args.override))
    OmegaConf.save(cfg, args.resolved)
    if (cfg.dino.loss_weight, cfg.ibot.loss_weight, cfg.dino.koleo_loss_weight,
        cfg.gram.use_loss, cfg.student.n_storage_tokens, cfg.student.mask_k_bias) != (1, 1, 0.1, False, 4, True):
        raise ValueError('B_full algorithm or architecture drift')
    if cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.optim.epochs != 100000:
        raise ValueError('Optimizer update budget is not exactly 100000')
    torch.manual_seed(0)
    with torch.device('meta'):
        net = SSLMetaArch(cfg)
    net.to_empty(device='cuda')
    state = torch.load(args.weight, map_location='cpu', mmap=True, weights_only=True)
    target = net.student.backbone.state_dict()
    missing = sorted(set(target) - set(state))
    unexpected = sorted(set(state) - set(target))
    shape_mismatch = {key: {'official': list(state[key].shape), 'student': list(target[key].shape)}
                      for key in set(target) & set(state) if state[key].shape != target[key].shape}
    if missing or unexpected or shape_mismatch:
        raise ValueError(f'Backbone load mismatch: missing={missing}, unexpected={unexpected}, shape={shape_mismatch}')
    net.student.backbone.load_state_dict(state, strict=True)
    net.student.dino_head.init_weights()
    net.student.ibot_head.init_weights()
    net.dino_loss.init_weights()
    net.ibot_patch_loss.init_weights()
    net.model_ema.load_state_dict(net.student.state_dict(), strict=True)
    with torch.device('meta'):
        official, _ = build_model_from_cfg(cfg, only_teacher=True)
    official.to_empty(device='cuda')
    official.load_state_dict(state, strict=True)
    official.eval()
    net.student.backbone.eval()
    torch.manual_seed(42)
    images = torch.rand(3, 3, 256, 256, device='cuda')
    metrics = {}
    with torch.inference_mode():
        a = official.forward_features(images)
        b = net.student.backbone.forward_features(images)
        for name, key in [('cls', 'x_norm_clstoken'), ('patch', 'x_norm_patchtokens')]:
            x, y = a[key].float(), b[key].float()
            delta = (x-y).abs()
            metrics[name] = {'shape': list(x.shape), 'official_norm': float(x.norm()),
                             'student_norm': float(y.norm()),
                             'cosine_similarity': float(F.cosine_similarity(x.flatten(), y.flatten(), dim=0)),
                             'max_absolute_difference': float(delta.max()),
                             'mean_absolute_difference': float(delta.mean())}
            if metrics[name]['cosine_similarity'] < 0.999999 or metrics[name]['max_absolute_difference'] > 1e-5:
                raise ValueError(f'Official/training output mismatch for {name}: {metrics[name]}')
    loaded_parameters = sum(value.numel() for value in state.values())
    total_parameters = sum(value.numel() for value in target.values())
    result = {'status': 'pass', 'official_weight_path': str(args.weight),
              'official_weight_sha256': actual_hash, 'file_size': args.weight.stat().st_size,
              'mtime': args.weight.stat().st_mtime, 'matched_keys': len(target),
              'missing_keys': missing, 'unexpected_keys': unexpected,
              'shape_mismatch': shape_mismatch,
              'loaded_parameter_count': loaded_parameters,
              'total_parameter_count': total_parameters,
              'coverage_ratio': loaded_parameters / total_parameters,
              'all_transformer_blocks_loaded': True,
              'patch_embed_loaded': True, 'normalization_loaded': True,
              'attention_mlp_loaded': True, 'feature_comparison': metrics,
              'teacher_initialized_from_student': True}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'pass', 'matched_keys': len(target), 'coverage_ratio': result['coverage_ratio'],
                      'feature_comparison': metrics}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
