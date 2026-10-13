set -euo pipefail
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import json,time,subprocess
r=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012')
target=r/'evaluation/comparison_100k_150k.json'
for i in range(40):
 status=json.loads((r/'STATUS.json').read_text())
 print('STATUS',json.dumps(status),flush=True)
 if target.exists():
  print('COMPARISON_COMPLETE',target.read_text(),flush=True)
  break
 for ds in ['Dataset111_CMCC','Dataset131_CMCT']:
  p=r/'evaluation'/(ds+'.log')
  if p.exists():print('LATEST',ds,' | '.join(p.read_text(errors='replace').splitlines()[-2:]),flush=True)
 if status.get('evaluation')=='failed':
  print(subprocess.run(['journalctl','-u','mitoverse-B-full-extend150k-20261012.service','-n','45','--no-pager'],capture_output=True,text=True).stdout,flush=True)
  raise SystemExit(2)
 time.sleep(30)
else:print('WATCH_TIMEOUT; EVALUATION_REMAINS_RUNNING',flush=True)
PY
