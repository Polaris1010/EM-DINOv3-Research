"""Read-only label and valid-region preflight for later full-volume supervision."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import zarr

from full_manifest import DATA_ROOT, sha256, validate_manifest


RUN = Path(__file__).resolve().parent.parent


def main() -> int:
    manifest = RUN / 'data/pretrain_all_mitoverse.json'
    audit = validate_manifest(manifest)
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    details = []
    errors = []
    for row in rows:
        vid = row['volume_id']
        path = Path(row['local_path'])
        try:
            group = zarr.open_group(path, mode='r')
            image, label = group['img'], group['mito']
            mask = group['mask'] if 'mask' in group else None
            if label.shape != image.shape or (mask is not None and mask.shape != image.shape):
                raise ValueError('image, label, or mask shape mismatch')
            if np.dtype(label.dtype).kind not in 'uib':
                raise ValueError(f'noninteger label dtype: {label.dtype}')
            if mask is not None and np.dtype(mask.dtype).kind not in 'uib':
                raise ValueError(f'noninteger mask dtype: {mask.dtype}')
            depth, height, width = image.shape
            rng = np.random.default_rng(int.from_bytes(vid.encode()[:8].ljust(8, b'\0'), 'little'))
            samples = []
            for sample_index in range(3):
                z = depth // 2 if sample_index == 0 else int(rng.integers(depth))
                y = max(0, (height - 256) // 2) if sample_index == 0 else int(rng.integers(max(1, height - 255)))
                x = max(0, (width - 256) // 2) if sample_index == 0 else int(rng.integers(max(1, width - 255)))
                label_crop = np.asarray(label[z, y:y + 256, x:x + 256])
                valid = (np.asarray(mask[z, y:y + 256, x:x + 256]) != 0
                         if mask is not None else np.ones(label_crop.shape, dtype=bool))
                if label_crop.size == 0 or not np.isfinite(label_crop).all():
                    raise ValueError('unreadable label crop')
                samples.append({'z': z, 'y': y, 'x': x, 'pixels': int(label_crop.size),
                                'valid_pixels': int(valid.sum()),
                                'foreground_valid_pixels': int(((label_crop > 0) & valid).sum())})
            metadata = {'mito/.zarray': sha256(path / 'mito/.zarray')}
            if mask is not None:
                metadata['mask/.zarray'] = sha256(path / 'mask/.zarray')
            details.append({'volume_id': vid, 'source': row['original_dataset_id'],
                            'label_shape_zyx': list(label.shape), 'label_dtype': str(label.dtype),
                            'mask_present': mask is not None,
                            'valid_policy': 'mask_nonzero' if mask is not None else 'all_pixels',
                            'metadata_sha256': metadata, 'sampled_crops': samples})
        except Exception as exc:
            errors.append({'volume_id': vid, 'error': repr(exc)})
    result = {'status': 'complete' if not errors else 'blocked',
              'scope': 'all_catalog_volumes', 'volume_count': len(details),
              'expected_count': audit['volume_count'],
              'by_source': dict(sorted(collections.Counter(row['source'] for row in details).items())),
              'mask_volume_count': sum(row['mask_present'] for row in details),
              'label_binary_rule': 'mito > 0',
              'valid_region_rule': 'mask != 0 where mask exists; all pixels otherwise',
              'sample_limit': 'Three 256x256 image locations per volume; not a full label scan',
              'errors': errors, 'volumes': details}
    target = RUN / 'evidence/supervised_label_metadata_audit.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'volume_count': len(details),
                      'mask_volume_count': result['mask_volume_count'],
                      'errors': len(errors)}))
    return 0 if not errors else 2


if __name__ == '__main__':
    raise SystemExit(main())
