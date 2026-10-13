set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
out="$run/runs/B_full/formal_attempt1"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json,os
from pathlib import Path
run=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
out=run/'runs/B_full/formal_attempt1'
p=json.loads((out/'progress.json').read_text())
latest=json.loads((out/'latest_checkpoint.json').read_text()) if (out/'latest_checkpoint.json').exists() else None
print('step',p['step'],'loss',p['loss'],'grad_norm',p['grad_norm'])
print('latest',latest)
if latest:
 cp=Path(latest['path'])
 print('checkpoint_exists',cp.exists(),'bytes',cp.stat().st_size if cp.exists() else None)
 assert cp.exists() and cp.stat().st_size>5_000_000_000
 assert latest['step']==1000
 assert p['step']>1000
PY
systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState -p MainPID -p Result --no-pager
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
