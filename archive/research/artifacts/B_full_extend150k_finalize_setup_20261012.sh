set -euo pipefail
cd /opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import json,py_compile
p=Path('scripts/gpu_guard.py');s=p.read_text()
s=s.replace("'stage': 'B_full_running' if code == 0 else 'B_full_stopped'","'stage': 'B_full_150k_complete' if code == 0 else 'B_full_stopped'")
s=s.replace("'topology_feasibility': 'in_preparation'","'topology_feasibility': 'paused; no method training'")
compile(s,str(p),'exec');p.write_text(s)
for p in ['scripts/train_B_full.py','scripts/posttrain_evaluation.py','src/frozen_probe_150k.py']:py_compile.compile(p,doraise=True)
plan=json.loads(Path('continuation_plan.json').read_text())
assert plan['start_step']==100000 and plan['target_step']==150000
other=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002/STATUS.json')
d=json.loads(other.read_text());d['B_full_extension']={'authorized_by_user':True,'from_step':100000,'to_step':150000,'run':'/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012','schedule':'constant original min LR 1e-6','status':'ready_after_resume_smoke'}
d['long_training']='B_FULL_150K_EXTENSION_ONLY_AUTHORIZED'
d['topology_or_masking_long_training']='PROHIBITED'
assert d['P'].startswith('PAUSED')
other.write_text(json.dumps(d,indent=2))
print('SETUP_VALIDATED; USER_AUTHORIZATION_RECORDED_FOR_B_FULL_ONLY')
PY
