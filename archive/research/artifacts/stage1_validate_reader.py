"""Read first, middle, and last image slices of every manifest volume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['training_role'] != 'ssl_full_mitoverse':
        raise ValueError('Unexpected manifest role')
    results = []
    for row in manifest['volumes']:
        vid = row['volume_id']
        image = zarr.open_group(row['local_path'], mode='r')[row['image_array_path']]
        if list(image.shape) != row['shape'] or str(image.dtype) != str(np.dtype(row['dtype'])):
            raise ValueError(f'Metadata mismatch: {vid}')
        depth, height, width = image.shape
        for z in sorted({0, depth // 2, depth - 1}):
            try:
                plane = np.asarray(image[z, :, :])
                if plane.shape != (height, width) or not np.isfinite(plane).all():
                    raise ValueError(f'Invalid plane shape or values: {plane.shape}')
                if height < 1 or width < 1:
                    raise ValueError('No legal spatial crop')
            except Exception as exc:
                failure = {'status': 'fail', 'volume_id': vid, 'slice_index': z,
                           'exception': f'{type(exc).__name__}: {exc}', 'completed_reads': len(results)}
                args.output.write_text(json.dumps(failure, indent=2) + '\n')
                raise RuntimeError(f'Image read failed: {vid} z={z}: {exc}') from exc
            results.append({'volume_id': vid, 'slice_index': z, 'shape': list(plane.shape),
                            'dtype': str(plane.dtype), 'finite': True, 'crop_possible': True})
    summary = {'status': 'pass', 'volume_count': len(manifest['volumes']), 'read_count': len(results),
               'reads': results}
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'reads'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
