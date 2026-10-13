"""1000-sample deterministic DINO loader smoke with label access forbidden."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dinov3-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=1000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.dinov3_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import zarr
    from dinov3.configs import get_default_config
    from ssl_full_data import Collate, Images

    cfg = get_default_config()
    cfg.crops.global_crops_size = 256
    cfg.crops.local_crops_size = 112
    cfg.crops.local_crops_number = 8
    cfg.crops.rgb_mean = [0.485, 0.456, 0.406]
    cfg.crops.rgb_std = [0.229, 0.224, 0.225]
    cfg.student.patch_size = 16
    torch.set_num_threads(2)
    calls = {'image': 0, 'forbidden': []}
    manifest_doc = json.loads(args.manifest.read_text(encoding='utf-8'))
    group_type = type(zarr.open_group(manifest_doc['volumes'][0]['local_path'], mode='r'))
    original = group_type.__getitem__

    def guarded_getitem(group, key):
        if key != 'img':
            calls['forbidden'].append(str(key))
            raise RuntimeError('SSL pipeline attempted to access labels or non-image array')
        calls['image'] += 1
        return original(group, key)

    group_type.__getitem__ = guarded_getitem
    try:
        dataset = Images(args.manifest, cfg, seed=0, start=0, count=args.samples)
        if dataset.slice_count != sum(r['shape'][0] for r in dataset.rows):
            raise AssertionError('Slice pool is incomplete')
        collate = Collate(cfg)
        chain = hashlib.sha256()
        source_counts: dict[str, int] = {}
        volume_ids = set()
        first = None
        for i in range(0, args.samples, 2):
            batch = collate([dataset[i], dataset[i + 1]])
            if list(batch['collated_global_crops'].shape) != [4, 3, 256, 256]:
                raise AssertionError('Unexpected global crop shape')
            if list(batch['collated_local_crops'].shape) != [16, 3, 112, 112]:
                raise AssertionError('Unexpected local crop shape')
            if not torch.isfinite(batch['collated_global_crops']).all():
                raise AssertionError('Nonfinite global crop')
            if not torch.isfinite(batch['collated_local_crops']).all():
                raise AssertionError('Nonfinite local crop')
            if first is None:
                first = batch['_sample_locations'][0]
            chain.update(batch['_sample_digest'].encode())
            for location in batch['_sample_locations']:
                source = location['source']
                source_counts[source] = source_counts.get(source, 0) + 1
                volume_ids.add(location['volume_id'])
        repeat = Images(args.manifest, cfg, seed=0, start=0, count=2)
        repeat_first = repeat.raw(0)[1]
        if repeat_first != first:
            raise AssertionError('Dataset is not deterministic at fixed seed/index')
        result = {'status': 'pass', 'samples': args.samples,
                  'slice_population': dataset.slice_count,
                  'source_counts': dict(sorted(source_counts.items())),
                  'sampled_volume_count': len(volume_ids),
                  'first_location': first, 'sample_chain_sha256': chain.hexdigest(),
                  'image_getter_calls': calls['image'],
                  'forbidden_getter_calls': calls['forbidden'],
                  'label_access': False}
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('source_counts', 'first_location')}))
    finally:
        group_type.__getitem__ = original
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
