set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
run=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
for rel in ('data/manifest_summary.json','audit/reader_validation.json','audit/formal_B_full_admission.json','audit/initialization_check.json','runs/B_full/formal_attempt1/result.json','topology_feasibility/mam_sam_vit_b/paired/ph_complete.json'):
 p=run/rel
 print('FILE',rel,'exists',p.exists())
 if p.exists():
  x=json.loads(p.read_text())
  if rel.endswith('formal_B_full_admission.json'):
   print('status',x.get('status'),'gates',len(x.get('gates',[])))
  else: print(json.dumps(x,ensure_ascii=False)[:2500])
PY
git -C "$run" rev-parse HEAD
git -C "$run" status --short | head -20
