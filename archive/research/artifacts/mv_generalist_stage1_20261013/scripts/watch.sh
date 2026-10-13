set -e
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import json,time
r=Path('/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013')
for _ in range(240):
 try:
  j=json.loads((r/'STATUS.json').read_text());print('STATUS',json.dumps(j),flush=True)
  for f in sorted((r/'runs').glob('*/seed*/metrics.json')):
   v=json.loads(f.read_text());print('DONE',v['model'],v['seed'],v['completed_steps'],v['validation']['source_means'],flush=True)
  if j['status'] in {'complete','failed'}:
   for p in sorted((r/'logs').glob('*.log')):print('LOG_TAIL',p.name,p.read_text(errors='replace')[-1800:],flush=True)
   break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(30)
PY
