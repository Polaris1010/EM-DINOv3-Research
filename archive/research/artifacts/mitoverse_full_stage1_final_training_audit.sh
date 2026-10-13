set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
out="$run/runs/B_full/formal_attempt1"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
run=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
out=run/'runs/B_full/formal_attempt1'
for name in ('result.json','latest_checkpoint.json','progress.json','initialization.json'):
 p=out/name
 print(name,'exists',p.exists())
 if p.exists(): print(json.dumps(json.loads(p.read_text()),indent=2))
c=json.loads((out/'latest_checkpoint.json').read_text())
assert c['step']==100000
assert Path(c['path']).is_file()
print('final_checkpoint_bytes',Path(c['path']).stat().st_size)
for name in ('student_backbone.pth','teacher_backbone.pth','metadata.json'):
 p=out/'checkpoints/step000100000'/name
 print(name,'exists',p.is_file(),'bytes',p.stat().st_size if p.exists() else None)
r=json.loads((out/'result.json').read_text())
assert r['status']=='complete' and r['completed_steps']==100000
assert r['max_rank_parameter_difference']<=1e-6
assert r['tracked_student_delta_l2']>0 and r['tracked_teacher_delta_l2']>0
print('verification','PASS')
PY
systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState -p MainPID -p Result -p ExecMainStatus --no-pager
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
