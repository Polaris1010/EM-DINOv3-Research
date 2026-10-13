"""Build and verify the complete, image-only MitoVerse SSL manifest."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


DATA_ROOT = Path('/opt/data/3_117/tanghr/MitoVerse_data')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(data_root: Path) -> tuple[dict, dict[str, Path]]:
    root = data_root.resolve(strict=True)
    catalog = json.loads((root / 'catalog.json').read_text(encoding='utf-8'))
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError('Catalog must be a nonempty mapping of volume IDs')
    actual = {str(path.resolve(strict=True)): path.resolve(strict=True)
              for path in (root / 'data').rglob('*.zarr') if path.is_dir()}
    if len(actual) != len(list((root / 'data').rglob('*.zarr'))):
        raise ValueError('Duplicate resolved Zarr directory path')
    expected = {}
    for vid, record in catalog.items():
        if not isinstance(record, dict) or not record.get('zarr') or not record.get('dataset_id'):
            raise ValueError(f'Incomplete catalog record: {vid}')
        path = (root / record['zarr']).resolve(strict=True)
        if not path.is_relative_to(root / 'data'):
            raise ValueError(f'Catalog path escapes data root: {vid}')
        if str(path) in expected:
            raise ValueError(f'Duplicate catalog path: {vid}')
        expected[str(path)] = vid
    if set(actual) != set(expected):
        raise ValueError(json.dumps({'catalog_only': sorted(set(expected) - set(actual)),
                                     'disk_only': sorted(set(actual) - set(expected))}, indent=2))
    return catalog, actual


def validate_manifest(manifest: Path, data_root: Path = DATA_ROOT, read_center: bool = False) -> dict:
    catalog, actual = inventory(data_root)
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Full SSL manifest must be a nonempty list')
    ids = [row['volume_id'] for row in rows]
    paths = [str(Path(row['local_path']).resolve(strict=True)) for row in rows]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError('Duplicate volume ID or path in full SSL manifest')
    if set(ids) != set(catalog) or set(paths) != set(actual):
        raise ValueError('Full SSL manifest differs from catalog or disk inventory')
    by_source = collections.Counter()
    for row, path_str in zip(rows, paths):
        vid = row['volume_id']
        item = catalog[vid]
        expected_path = str((data_root.resolve() / item['zarr']).resolve(strict=True))
        if path_str != expected_path or row['original_dataset_id'] != item['dataset_id']:
            raise ValueError(f'Catalog mapping mismatch: {vid}')
        if row.get('training_role') != 'ssl_full_mitoverse' or row.get('axes') != 'ZYX':
            raise ValueError(f'Invalid training role or axes: {vid}')
        if row.get('legacy_split') is None:
            raise ValueError(f'Missing legacy split trace: {vid}')
        path = Path(path_str)
        meta = path / 'img' / '.zarray'
        info = json.loads(meta.read_text(encoding='utf-8'))
        if list(info['shape']) != row['image_shape_zyx'] or info['dtype'] != row['image_dtype']:
            raise ValueError(f'Image metadata changed: {vid}')
        for relative, expected_hash in row['metadata_sha256'].items():
            if sha256(path / relative) != expected_hash:
                raise ValueError(f'Metadata hash changed: {vid} {relative}')
        if read_center:
            import numpy as np
            import zarr
            image = zarr.open_group(path, mode='r')['img']
            depth, height, width = image.shape
            sample = np.asarray(image[depth // 2, height // 2:height // 2 + 1,
                                      width // 2:width // 2 + 1])
            if sample.size != 1 or not np.isfinite(sample).all():
                raise ValueError(f'Unreadable image center: {vid}')
        by_source[row['original_dataset_id']] += 1
    return {'volume_count': len(rows), 'source_count': len(by_source),
            'by_source': dict(sorted(by_source.items())), 'manifest_sha256': sha256(manifest),
            'catalog_sha256': sha256(data_root / 'catalog.json'),
            'catalog_disk_manifest_paths_equal': True,
            'volume_ids': sorted(ids)}


def build(data_root: Path, old_manifest: Path, output: Path) -> dict:
    catalog, _ = inventory(data_root)
    legacy = {}
    with old_manifest.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            vid = row['volume_id']
            if vid in legacy:
                raise ValueError(f'Duplicate legacy ID: {vid}')
            legacy[vid] = row['split']
    rows = []
    for vid, item in sorted(catalog.items()):
        path = (data_root / item['zarr']).resolve(strict=True)
        meta = path / 'img' / '.zarray'
        info = json.loads(meta.read_text(encoding='utf-8'))
        shape = info['shape']
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError(f'Invalid image shape: {vid} {shape}')
        if info['dtype'] not in {'|u1', '<u1', '>u1'}:
            raise ValueError(f'Image is not uint8 under the locked normalization: {vid} {info["dtype"]}')
        zattrs = path / '.zattrs'
        if not zattrs.is_file():
            raise ValueError(f'Missing volume metadata: {vid}')
        rows.append({'volume_id': vid, 'original_dataset_id': item['dataset_id'],
                     'local_path': str(path), 'image_shape_zyx': shape,
                     'image_dtype': info['dtype'], 'axes': 'ZYX',
                     'legacy_split': legacy.get(vid, 'not_in_legacy_manifest'),
                     'training_role': 'ssl_full_mitoverse',
                     'metadata_sha256': {'img/.zarray': sha256(meta),
                                         '.zattrs': sha256(zattrs)}})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output)
    return validate_manifest(output, data_root, read_center=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--old-manifest', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    result = (build(args.data_root, args.old_manifest, args.manifest)
              if args.old_manifest else validate_manifest(args.manifest, args.data_root, read_center=True))
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'volume_ids'}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
