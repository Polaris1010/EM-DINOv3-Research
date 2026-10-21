"""Isolated fixed-split decoder-seed replication; historical assets are read-only."""
import hashlib
import json
import os
from pathlib import Path

R = Path(os.environ.get('D12_SEEDS_ROOT', '/opt/data/3_117/tanghr/codex/d12_three_seeds_20261022'))
M3 = R.parent / 'mv_generalist_v2_han_fullgt_20261014'
OLD = R.parent / 'mv_generalist_stage1_20261013'
MV2 = R.parent / 'mv_dev_v2_notta_20261013'
PY = R.parent.parent / 'envs/meddinov3/bin/python'
NR = R.parent / 'coverage_nr_p0_20261021'

def read(p):
    return json.loads(Path(p).read_text())

def atomic(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    t.replace(p)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def heartbeat(path):
    import time
    p = Path(path)
    t = p.with_name(p.name + '.seed-supervisor.tmp')
    t.write_text(str(time.time()))
    t.replace(p)
