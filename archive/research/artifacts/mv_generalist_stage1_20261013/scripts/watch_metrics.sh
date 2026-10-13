set -e
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import json,time
r=Path('/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013');previous=None
for _ in range(320):
 try:
  j=json.loads((r/'STATUS.json').read_text())
  parts=[j['status'],j.get('stage')]
  for w in j.get('workers',[]):
   p=w.get('progress') or {};e=w.get('eval_progress') or {}
   parts.append({k:p.get(k) for k in ['model','seed','step']})
   if e:parts.append({k:e.get(k) for k in ['volume','z_completed']})
  if j['status'].startswith('scoring'):
   f=r/'logs'/('single_seed_analysis.log' if j['status']=='scoring_single_seed' else 'analysis.log')
   if f.exists():parts.extend(s for s in f.read_text(errors='replace').splitlines() if s.startswith(('SCORED','REUSED','MV_ANALYSIS')))
  s=json.dumps(parts)
  if s!=previous:print(s,flush=True);previous=s
  if j['status'] in {'complete','failed'}:print(json.dumps(j),flush=True);break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(45)
PY
