"""Freeze coordinates before features; evaluation labels and model failures are not consulted."""
import collections
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
from common import R, MV, MB, read, atomic, sha, manifest, cache, prime_prediction


def make():
    protocol = read(R/'protocol.json')
    assert not (R/'samples.json').exists(), 'Never silently resample a locked diagnostic'
    rng = np.random.default_rng(protocol['seed'])
    rows = manifest()['entries']
    train = [r for r in rows if r['supervised_role'] == 'train']
    assert len(train) == 156
    items = []

    def add(row, domain, role, z, y, x, subset):
        if domain == 'MV':
            shape = cache(row)['shape_zyx']
            spacing = row['spacing_zyx_nm'][-2:]
            offsets = [r[0] for r in row['region_zyx']]
        else:
            shape, spacing = row['shape_zyx'], row['spacing_yx_nm']
            offsets = [0, 0, 0]
        source = row.get('source_id', row.get('dataset'))
        item = dict(id=len(items), domain=domain, role=role, subset=subset, source=source,
                    region_id=row['id'], volume_id=row.get('volume_id', row['id']),
                    acquisition=row.get('acquisition_id') or row.get('volume_id', row['id']),
                    z=int(z), y=int(y), x=int(x), shape_zyx=shape, spacing_yx_nm=spacing,
                    native_zyx=[int(z+offsets[0]), int(y+offsets[1]), int(x+offsets[2])])
        items.append(item)

    def random_position(row, domain, used):
        if domain == 'MV':
            c = cache(row)
            shape, eligible = c['shape_zyx'], c['valid_z']
        else:
            shape, eligible = row['shape_zyx'], range(row['shape_zyx'][0])
        for _ in range(10000):
            z = int(eligible[int(rng.integers(len(eligible)))])
            y, x = [int(rng.integers(max(1, n-256+1))) for n in shape[1:]]
            key = row['id'], z, y, x
            if key not in used:
                used.add(key)
                return z, y, x
        raise ValueError('Insufficient unique positions for fixed budget')

    used = set()
    for row in sorted(train, key=lambda r: r['id']):
        for _ in range(protocol['train_crops_per_region']):
            add(row, 'MV', 'train', *random_position(row, 'MV', used), 'train')
    for source in protocol['diagnostic_sources']:
        domain = 'MB' if source in ('glycolytic_muscle', 'hela_cell') else 'MV'
        pool = [r for r in rows if r['supervised_role']=='dev' and r['source_id']==source] if domain=='MV' else [r for r in read(MB/'MB1_benchmark_manifest.json')['entries'] if r['dataset']==source]
        assert pool
        pool = sorted(pool, key=lambda r: r['id'])
        order = rng.permutation(len(pool))
        for k in range(protocol['random_crops_per_source']):
            row = pool[int(order[k % len(pool)])]
            add(row, domain, 'eval', *random_position(row, domain, used), 'random')
    row = next(r for r in read(MB/'MB1_benchmark_manifest.json')['entries'] if r['id']=='glycolytic_muscle')
    case = protocol['historical_error_case']
    add(row, 'MB', 'eval', case['z'], case['y'], case['x'], 'historical_D2_error_case')
    assert {r['region_id'] for r in items if r['role']=='train'} == {r['id'] for r in train}
    assert len({(r['domain'], r['region_id'], r['z'], r['y'], r['x']) for r in items}) == len(items)
    atomic(R/'samples.json', items)
    atomic(R/'sampling_receipt.json', {'status':'LOCKED_BEFORE_FEATURES', 'seed':protocol['seed'],
           'samples_sha256':sha(R/'samples.json'), 'protocol_sha256':sha(R/'protocol.json'),
           'by_role':dict(collections.Counter(r['role'] for r in items)),
           'random_by_source':dict(collections.Counter(r['source'] for r in items if r['subset']=='random')),
           'train_regions':156,'evaluation_GT_read':False,'prediction_selection_used':False,
           'historical_case_excluded_from_random':True,'created_at':time.time()})
    models=[]
    for row in protocol['models']:
        path=Path(row['checkpoint_path']);assert path.is_file() and sha(path)==row['sha256']
        assert sha(row['resolved_config_path'])==row['resolved_config_sha256']
        stat=path.stat();models.append({**row,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    atomic(R/'verified_models.json',models)
    atomic(R/'evidence/environment.json',{'python':platform.python_version(),'executable':__import__('sys').executable,
           'platform':platform.platform(),'code_commit':(R/'CODE_COMMIT').read_text().strip(),
           'source_sha256':{p.name:sha(p) for p in (R/'code').glob('*.py')},
           'manifest_sha256':sha(MV/'MV-Generalist.json')})
    return items


if __name__=='__main__':
    make()
