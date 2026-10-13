"""Build a fresh, image-only Full MitoVerse SSL manifest from catalog and disk."""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-csv', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    catalog_path = root / 'catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError('Expected a nonempty catalog mapping')
    disk = {p.resolve(strict=True): p for p in (root / 'data').rglob('*.zarr') if p.is_dir()}
    expected: dict[Path, str] = {}
    for vid, item in catalog.items():
        if item.get('volume_id') != vid or not item.get('dataset_id') or not item.get('zarr'):
            raise ValueError(f'Incomplete catalog record: {vid}')
        path = (root / item['zarr']).resolve(strict=True)
        if not path.is_relative_to(root / 'data'):
            raise ValueError(f'Escaping catalog path: {vid}')
        if path in expected:
            raise ValueError(f'Duplicate catalog path: {vid}, {expected[path]}')
        expected[path] = vid
    if set(disk) != set(expected):
        raise ValueError(json.dumps({'catalog_only': sorted(map(str, set(expected) - set(disk))),
                                     'disk_only': sorted(map(str, set(disk) - set(expected)))}, indent=2))

    legacy: dict[str, str] = {}
    with args.legacy_csv.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['volume_id'] in legacy:
                raise ValueError(f'Duplicate legacy ID: {row["volume_id"]}')
            legacy[row['volume_id']] = row['split']
    if set(legacy) != set(catalog):
        raise ValueError('Legacy trace ID set differs from current catalog; investigate before recording splits')

    rows = []
    inventory = []
    by_source: collections.Counter[str] = collections.Counter()
    slice_count = 0
    for vid, item in sorted(catalog.items()):
        path = (root / item['zarr']).resolve(strict=True)
        group_meta = path / '.zgroup'
        attrs_path = path / '.zattrs'
        image_meta = path / 'img' / '.zarray'
        for meta_path in (group_meta, attrs_path, image_meta):
            if not meta_path.is_file():
                raise ValueError(f'Missing Zarr metadata: {vid} {meta_path}')
        info = json.loads(image_meta.read_text(encoding='utf-8'))
        attrs = json.loads(attrs_path.read_text(encoding='utf-8'))
        shape = info['shape']
        chunks = info['chunks']
        if len(shape) != 3 or min(shape) < 1 or len(chunks) != 3:
            raise ValueError(f'Invalid ZYX image array: {vid}: {shape}, {chunks}')
        if shape != item.get('shape_zyx') or info['dtype'] not in ('|u1', '<u1', '>u1'):
            raise ValueError(f'Unexpected image metadata: {vid}: {shape}, {info["dtype"]}')
        if attrs.get('mitoverse', {}).get('volume_id') != vid:
            raise ValueError(f'Zarr group identity mismatch: {vid}')
        source = item['dataset_id']
        row = {
            'volume_id': vid,
            'original_dataset_id': source,
            'local_path': str(path),
            'image_array_path': 'img',
            'shape': shape,
            'dtype': info['dtype'],
            'axis_order': 'ZYX',
            'chunks': chunks,
            'voxel_size': item.get('voxel_nm'),
            'voxel_size_unit': 'nm',
            'source_dataset': source,
            'legacy_split': legacy[vid],
            'training_role': 'ssl_full_mitoverse',
            'metadata_sha256': {
                '.zgroup': digest(group_meta),
                '.zattrs': digest(attrs_path),
                'img/.zarray': digest(image_meta),
            },
        }
        rows.append(row)
        inventory.append({'volume_id': vid, 'source': source, 'local_path': str(path),
                          'number_of_slices': shape[0], 'legacy_split': legacy[vid]})
        by_source[source] += 1
        slice_count += shape[0]

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run = output.parent.parent
    manifest = {
        'manifest_version': 'full_mitoverse_ssl_v1',
        'generated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'catalog_path': str(catalog_path),
        'catalog_sha256': digest(catalog_path),
        'volume_count': len(rows),
        'source_count': len(by_source),
        'training_role': 'ssl_full_mitoverse',
        'volumes': rows,
    }
    write_json(output, manifest)
    manifest_hash = digest(output)
    (output.parent / 'pretrain_all_mitoverse.sha256').write_text(manifest_hash + '  ' + output.name + '\n')
    write_json(output.parent / 'manifest_summary.json', {
        'volume_count': len(rows), 'source_count': len(by_source),
        'slice_count': slice_count, 'source_counts': dict(sorted(by_source.items())),
        'manifest_sha256': manifest_hash, 'catalog_sha256': digest(catalog_path),
        'catalog_disk_manifest_paths_equal': True,
    })
    write_json(run / 'audit' / 'source_counts.json', dict(sorted(by_source.items())))
    with (run / 'audit' / 'volume_inventory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    with (run / 'audit' / 'sampling_distribution.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['volume_id', 'number_of_slices', 'fraction_of_slice_pool', 'source'])
        writer.writeheader()
        for entry in inventory:
            writer.writerow({'volume_id': entry['volume_id'], 'number_of_slices': entry['number_of_slices'],
                             'fraction_of_slice_pool': entry['number_of_slices'] / slice_count,
                             'source': entry['source']})
    print(json.dumps({'volume_count': len(rows), 'source_count': len(by_source),
                      'slice_count': slice_count, 'manifest_sha256': manifest_hash,
                      'by_source': dict(sorted(by_source.items()))}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
