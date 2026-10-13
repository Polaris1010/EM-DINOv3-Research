set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
run=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
out=run/'runs/B_full/formal_attempt1'
p=json.loads((out/'progress.json').read_text())
c=json.loads((out/'latest_checkpoint.json').read_text())
print('step',p['step'],'percent',round(p['step']/1000,2),'elapsed_hours',round(p['elapsed_seconds']/3600,2),'loss',p['loss'],'grad_norm',p['grad_norm'])
print('checkpoint_step',c['step'],'checkpoint_sha256',c['sha256'],'exists',Path(c['path']).exists())
print('remaining_hours_at_observed_average',round((100000-p['step'])*(p['elapsed_seconds']/p['step'])/3600,1))
PY
systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState -p Result --no-pager
date -Is
