"""Fail-closed evidence gate for the four new 100k-step SSL runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from full_manifest import validate_manifest, sha256


RUN = Path(__file__).resolve().parent.parent
WEIGHT = Path('/opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
WEIGHT_SHA = '8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035'
ARMS = ('B', 'L', 'T', 'LT')
METHODS = ('train_ssl.py', 'ssl_data.py', 'full_manifest.py', 'ssl_base_config.yaml',
           'last_selection.py', 'topology_appearance.py')


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def assess(arm: str, kind: str, count: int) -> dict:
    job = RUN / 'src/jobs' / f'{arm}_seed0_batch16_{kind}{count}'
    result = load(job / 'result.json')
    init = load(job / 'initialization.json')
    steps = [json.loads(line) for line in (job / 'steps.jsonl').read_text().splitlines()]
    if (result['status'] != 'complete' or result['completed_steps'] != count
            or init['source'] != 'official_natural_dinov3_vitl16'
            or init['weight_sha256'] != WEIGHT_SHA or init['resume'] is not None
            or init['starting_step'] != 0 or not init['strict_backbone_load']
            or init['missing_keys'] or init['unexpected_keys']
            or len(steps) != count or [row['step'] for row in steps] != list(range(1, count + 1))):
        raise ValueError(f'Incomplete or incorrectly initialized {arm} {kind} evidence')
    if any(not math.isfinite(row['loss']) or not math.isfinite(row['grad_norm'])
           or row['grad_norm'] <= 0 or row['feature_stat']['cls_batch_std_mean'] <= 1e-4
           or row['feature_stat']['patch_spatial_std_mean'] <= 1e-4 for row in steps):
        raise ValueError(f'Unstable {arm} {kind} evidence')
    if not any(value > 0 for value in steps[-1]['tracked_parameter_delta_l2'].values()):
        raise ValueError(f'No student parameter update for {arm} {kind}')
    progress = load(job / 'progress.json')
    config = OmegaConf.load(job / 'resolved_config.yaml')
    if (config.student.pretrained_weights or config.teacher.pretrained_weights
            or config.student.arch != 'vit_large' or config.student.n_storage_tokens != 4
            or not config.student.mask_k_bias or config.train.batch_size_per_gpu != 16
            or config.train.dataset_path != str(RUN / 'data/pretrain_all_mitoverse.json')
            or config.train.output_dir != str(RUN / 'jobs')
            or config.crops.global_crops_size != 256
            or config.crops.local_crops_size != 112
            or config.crops.local_crops_number != 8):
        raise ValueError(f'Unexpected resolved config for {arm} {kind}')
    if sum(progress['source_exposure'].values()) != count * 16:
        raise ValueError(f'Incomplete sample exposure for {arm} {kind}')
    if sum(progress['volume_exposure'].values()) != count * 16:
        raise ValueError(f'Incomplete volume exposure for {arm} {kind}')
    if set(progress['source_exposure']) != set(load(RUN / 'evidence/full_manifest_audit.json')['by_source']):
        raise ValueError(f'Not all sources sampled in {arm} {kind}')
    return {'arm': arm, 'phase': kind, 'steps': count, 'sample_chain': result['sample_chain'],
            'peak_gpu_bytes': max(row['peak_gpu_bytes'] for row in steps),
            'min_grad_norm': min(row['grad_norm'] for row in steps),
            'min_cls_std': min(row['feature_stat']['cls_batch_std_mean'] for row in steps),
            'min_patch_std': min(row['feature_stat']['patch_spatial_std_mean'] for row in steps),
            'covered_volumes': len(progress['volume_exposure'])}


def main() -> int:
    manifest = RUN / 'data/pretrain_all_mitoverse.json'
    audit = validate_manifest(manifest, read_center=True)
    if sha256(WEIGHT) != WEIGHT_SHA:
        raise ValueError('Official weight SHA256 mismatch')
    topology = load(RUN / 'topology_calibration/topology_selected_locked.json')
    if (topology['status'] != 'complete' or topology['manifest_sha256'] != audit['manifest_sha256']
            or topology['source_count'] != audit['source_count']
            or topology['volume_count'] != audit['volume_count']):
        raise ValueError('Topology selection is not locked to the full manifest')
    if load(RUN / 'evidence/resume_sequence_check.json').get('status') != 'passed':
        raise ValueError('New-run sample sequence resume check is missing')
    rows = [assess(arm, kind, count) for kind, count in (('smoke', 10), ('stability', 100))
            for arm in ARMS]
    for kind in ('smoke', 'stability'):
        chains = {row['sample_chain'] for row in rows if row['phase'] == kind}
        if len(chains) != 1:
            raise ValueError(f'Four arms have different sample sequences in {kind}')
    hashes = {name: sha256(RUN / 'src' / name) for name in METHODS}
    hashes['pretrain_all_mitoverse.json'] = sha256(manifest)
    hashes['topology_selected_locked.json'] = sha256(RUN / 'topology_calibration/topology_selected_locked.json')
    result = {'status': 'admitted', 'data_scope': 'all_mitoverse',
              'initialization': 'official_natural_vitl16', 'resume': None,
              'batch': 16, 'target_steps_each': 100000, 'seed': 0,
              'authorized_gpus': [1, 2, 3], 'forbidden_gpu': 0,
              'manifest_audit': audit, 'natural_weight_sha256': WEIGHT_SHA,
              'method_hashes': hashes, 'validation': rows}
    path = RUN / 'evidence/formal_seed0_admission.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'manifest_sha256': audit['manifest_sha256'],
                      'volume_count': audit['volume_count'], 'source_count': audit['source_count']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
