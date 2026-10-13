set -euo pipefail
TASK=/opt/data/3_117/tanghr/codex/mv_dev_v2_expansion_20261013
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import json,collections,subprocess,urllib.request
r=Path('/opt/data/3_117/tanghr/codex/mv_dev_v2_expansion_20261013')
for name in ['STATUS.json','MV_dev_validation_manifest.json']:
 p=r/name
 if p.exists():
  j=json.loads(p.read_text())
  if 'manifest' in name:print('COHORT',len(j['entries']),dict(collections.Counter(v['source_id'] for v in j['entries'])), 'test_reserved',len(j['reserved_test_ids']))
  else:print('STATUS',json.dumps(j))
for p in (r/'logs').glob('*.log'):
 with p.open() as f:lines=collections.deque(f,maxlen=5)
 print('LOG',p.name,''.join(lines))
print('GPU',subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
print('SITE',urllib.request.urlopen('http://127.0.0.1:7860/',timeout=8).status)
PY
