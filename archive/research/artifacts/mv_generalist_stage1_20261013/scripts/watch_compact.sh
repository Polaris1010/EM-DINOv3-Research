set -e
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import json,time
r=Path('/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013');done=set()
for _ in range(240):
 try:
  j=json.loads((r/'STATUS.json').read_text());ws=[]
  for w in j.get('workers',[]):
   p=w.get('progress') or {};e=w.get('eval_progress') or {}
   ws.append({'model':p.get('model'),'seed':p.get('seed'),'step':p.get('step'),'loss':p.get('loss_median200'),'evaluation':e.get('volume'),'z':e.get('z_completed')})
  print('PROGRESS',json.dumps({'status':j['status'],'stage':j.get('stage'),'workers':ws}),flush=True)
  for f in sorted((r/'runs').glob('*/seed*/metrics.json')):
   if str(f) in done:continue
   v=json.loads(f.read_text());print('COMPLETED',v['model'],v['seed'],v['completed_steps'],v['validation']['source_means'],flush=True);done.add(str(f))
  if j['status'] in {'complete','failed'}:
   for p in sorted((r/'logs').glob('*.log')):print('LOG_TAIL',p.name,p.read_text(errors='replace')[-1000:],flush=True)
   break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(45)
PY
