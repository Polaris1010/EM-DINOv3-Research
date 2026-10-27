"""Pilot A I/O and immutable references; historical assets are always opened read-only."""
import ast
import hashlib
import json
import os
from pathlib import Path
import numpy as np

R = Path(os.environ.get('PILOT_A_ROOT', '/opt/data/3_117/tanghr/codex/feature_mechanism_pilot_a_20261027'))
BASE = R.parent
MV = BASE/'coverage_nr_p0_20261021/downstream'
MV2 = BASE/'mv_dev_v2_notta_20261013'
MB = BASE/'mb1_official_benchmark_20261015'
PY = '/opt/data/3_117/tanghr/envs/meddinov3/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def npy(path):
    try:
        return np.load(path, mmap_mode='r')
    except ValueError:
        # Historical NumPy wrote a list shape. Read its header without modifying the file.
        with Path(path).open('rb') as handle:
            version = np.lib.format.read_magic(handle)
            length_bytes = 2 if version == (1, 0) else 4
            size = int.from_bytes(handle.read(length_bytes), 'little')
            header = ast.literal_eval(handle.read(size).decode('latin1').strip())
            offset = handle.tell()
        return np.memmap(path, dtype=np.dtype(header['descr']), mode='r', offset=offset,
                         shape=tuple(header['shape']), order='F' if header['fortran_order'] else 'C')


def manifest():
    return read(MV/'MV-Generalist.json')


def cache(row):
    receipt = read(MV/'cache'/row['id']/'CACHE.json')
    assert receipt['manifest_sha256'] == sha(MV/'MV-Generalist.json')
    return receipt


class Data:
    def __init__(self):
        self.rows = {r['id']: r for r in manifest()['entries']}
        self.mb = {r['id']: r for r in read(MB/'MB1_benchmark_manifest.json')['entries']}
        self.arrays = {}
        self.files = []

    def plane(self, item, kind='image'):
        assert kind in ('image', 'label', 'valid')
        if item['domain'] == 'MV':
            row = self.rows[item['region_id']]
            c = cache(row)
            if kind == 'valid' and not c['valid_path']:
                return np.ones(tuple(c['shape_zyx'][1:]), bool)
            path = c['valid_path'] if kind == 'valid' else c['paths'][kind]
            if path not in self.arrays:
                self.arrays[path] = npy(path)
            if kind == 'valid':
                return self.arrays[path][item['z']].astype(bool)
            offsets = c['offsets']
            return self.arrays[path][item['z']+offsets[0][0], slice(*offsets[1]), slice(*offsets[2])]
        row = self.mb[item['region_id']]
        if kind == 'valid':
            return np.ones(tuple(row['shape_zyx'][1:]), bool)
        path = row['image_path'] if kind == 'image' else row['label_path']
        if path not in self.arrays:
            import tifffile
            try:
                self.arrays[path] = tifffile.memmap(path, mode='r')
            except (ValueError, TypeError):
                self.arrays[path] = tifffile.imread(path)
        return self.arrays[path][item['z']]

    def tile(self, item, kind='image'):
        plane = self.plane(item, kind)
        tile = np.asarray(plane[item['y']:item['y']+256, item['x']:item['x']+256], dtype=np.float32 if kind=='image' else None)
        pad = ((0, 256-tile.shape[0]), (0, 256-tile.shape[1]))
        return np.pad(tile, pad, constant_values=127.5 if kind == 'image' else 0)


def prime_prediction(item, model):
    """Locate only existing registered inference outputs, never synthesize a head."""
    roots = {
        'P0': ('mv_generalist_v2_han_fullgt_20261014', 'G-P0', 'mb1_official_benchmark_20261015', 'G-P0_seed20261014'),
        'D1-100k': ('ssl_diversity_d12_100k_20261018/downstream', 'D1', 'ssl_diversity_d12_100k_20261018', 'D1'),
        'D2-30k': ('ssl_diversity_d12_20261017/downstream', 'D2', 'ssl_diversity_d12_20261017', 'D2'),
        'D2-100k': ('ssl_diversity_d12_100k_20261018/downstream', 'D2', 'ssl_diversity_d12_100k_20261018', 'D2'),
        'Coverage-NR-100k': ('coverage_nr_p0_20261021/downstream', 'Coverage-NR', 'coverage_nr_p0_20261021', 'Coverage-NR'),
    }
    devroot, devmodel, mbroot, mbmodel = roots[model]
    if item['domain'] == 'MV':
        path = BASE/devroot/'runs'/devmodel/'seed20261014'/(item['region_id']+'_prediction.npy')
    else:
        path = BASE/mbroot/'benchmark'/mbmodel/(item['region_id']+'_prediction.npy')
        if model == 'P0':
            # MB1 registry supplies the actual run directory; no guessed replacement inference.
            protocol = read(MB/'MB1_protocol.json')
            run = next(r for r in protocol['runs'] if r.get('model') in ('G-P0', 'P0') and r.get('seed') == 20261014)
            path = MB/'runs'/run['run_id']/(item['region_id']+'_prediction.npy')
    return path
