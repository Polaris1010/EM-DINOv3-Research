set -e
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import psutil,shutil,json,subprocess
r=Path('/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013')
print('FREE_DISK_GIB',shutil.disk_usage(r).free/(1024**3))
print('AVAILABLE_RAM_GIB',psutil.virtual_memory().available/(1024**3))
print('CPU_COUNT',psutil.cpu_count())
print('CPU_PERCENT',psutil.cpu_percent(interval=1))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
for f in ['STATUS.json','pilot/P0/curve.json','MV_frozen_probe_protocol.yaml']:
 p=r/f
 if p.exists():
  j=json.loads(p.read_text())
  if f.endswith('curve.json'):j=[{'step':a['step'],'macro':a['descriptive_all_source_macro_dice'],'by_source':a['source_means'],'loss_median200':a['loss_median200']} for a in j]
  if f.endswith('.yaml'):j={k:j[k] for k in ['status','locked_steps','formal_seed_runs_may_start']}
  print(f,json.dumps(j))
PY
