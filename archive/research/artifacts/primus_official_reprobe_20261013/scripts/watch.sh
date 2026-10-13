set -euo pipefail
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import json,time,subprocess
r=Path('/opt/data/3_117/tanghr/codex/primus_official_reprobe_20261013')
last=None
if not (r/'STATUS.json').exists():
 print('SMOKE LOG', (r/'audit/smoke.log').read_text(),flush=True)
 raise SystemExit(2)
for i in range(180):
 try:s=json.loads((r/'STATUS.json').read_text())
 except (OSError,json.JSONDecodeError):s={'status':'starting'}
 rows=[]
 for w in s.get('workers',[]):
  rows.append({'model':w['model'],'exit_code':w['exit_code'],'runs':[{'dataset':x['dataset'].split('_')[-1],'step':(x['progress'] or {}).get('step',0),'complete':x['complete']} for x in w['runs']]})
 compact={'status':s['status'],'workers':rows}
 if (r/'SCORING_PROGRESS.json').exists():
  try:compact['scoring']=json.loads((r/'SCORING_PROGRESS.json').read_text())['completed_groups']
  except (OSError,json.JSONDecodeError):pass
 if compact!=last:print(json.dumps(compact),flush=True);last=compact
 active=subprocess.run(['systemctl','show','primus-official-reprobe-20261013.service','-p','ActiveState','-p','Result','--no-pager'],capture_output=True,text=True).stdout
 if s['status']=='complete' and (r/'delivery.tar.gz').exists() and 'ActiveState=active' not in active:
  print('DELIVERY READY',str(r/'delivery.tar.gz'),flush=True);break
 if s['status']=='failed' or 'ActiveState=failed' in active:
  print('FAILURE',json.dumps(s),flush=True)
  print(subprocess.run(['journalctl','-u','primus-official-reprobe-20261013.service','-n','30','--no-pager'],capture_output=True,text=True).stdout,flush=True)
  for p in sorted((r/'runs').glob('*.log')):print(p.name,p.read_text()[-2200:],flush=True)
  raise SystemExit(2)
 time.sleep(20)
else:print('MONITOR LIMIT REACHED',flush=True)
PY
