set -euo pipefail
env PYTHONPATH=/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
import json,collections
from pathlib import Path
import zarr
r=Path('/opt/data/3_117/tanghr/codex/representation_diagnosis_live_20261013')
vols=json.loads((r/'evidence/formal_pretrain_manifest.json').read_text())['volumes']
print('SPLITS',dict(collections.Counter(v['legacy_split'] for v in vols)),flush=True)
choices=[v for v in vols if 'train' in v['legacy_split'] and min(v['shape'][1:])>=512]
for v in choices[:8]:
 g=zarr.open(v['local_path'],mode='r');print('CANDIDATE',json.dumps(v), 'KEYS',list(g.keys()),'ATTRS',dict(g.attrs),'LABEL_ATTRS',dict(g['lbl'].attrs) if 'lbl' in g else None,flush=True)
(r/'evidence/roi_candidates.json').write_text(json.dumps(choices,indent=2))
PY
