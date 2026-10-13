set -e
systemctl show mv-generalist-stage1-20261013.service -p ActiveState -p MainPID -p Result --no-pager
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import json,subprocess,urllib.request
r=Path('/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013')
for name in ['STATUS.json','pilot/P0/progress.json','pilot/P0/curve.json','evidence/multi_source_smoke.json','evidence/single_seed_result_audit.json','analysis_completion.json']:
 p=r/name
 if p.exists():
  j=json.loads(p.read_text())
  if name=='evidence/multi_source_smoke.json':j={'status':j['status'],'models':[{k:v for k,v in m.items() if k in ['model','status','peak_memory_bytes','encoder_unchanged']} for m in j['models']]}
  print(name,json.dumps(j))
for p in sorted((r/'logs').glob('*.log')):
 text=p.read_text(errors='replace');print('LOG_TAIL',p.name,text[-1300:])
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
print('SITE',urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10).status)
PY
