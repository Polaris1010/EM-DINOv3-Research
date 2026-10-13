set -e
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import psutil,json,subprocess
from pathlib import Path
r=Path('/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013')
print('SERVER_TIME',__import__('datetime').datetime.now().isoformat())
import importlib.util
print('CUPY_AVAILABLE',importlib.util.find_spec('cupy') is not None)
items=[]
for p in psutil.process_iter(['pid','cmdline','memory_info','cpu_times','create_time']):
 try:
  cmd=' '.join(p.info['cmdline'] or [])
  if str(r) in cmd:
   items.append({'pid':p.pid,'rss_gib':round(p.info['memory_info'].rss/2**30,2),'cpu_s':round(sum(p.info['cpu_times'][:2]),1),'elapsed_s':round(__import__('time').time()-p.info['create_time']),'command':cmd[-180:]})
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
print(json.dumps(items));print('available_gib',psutil.virtual_memory().available/2**30)
for item in items:
 if 'scripts/analyze.py' in item['command']:
  for p in psutil.Process(item['pid']).children(recursive=True):
   try:
    c=p.cpu_times();print('SCORING_CHILD',p.pid,'rss_gib',round(p.memory_info().rss/2**30,2),'cpu_s',round(c.user+c.system,1),'cpu_percent',p.cpu_percent(interval=1),'status',p.status())
   except psutil.NoSuchProcess:pass
for f in sorted((r/'runs').glob('*/seed*/metrics.json')):
 v=json.loads(f.read_text());print('FULL_DICE',v['model'],v['seed'],v['validation']['source_means'])
log=r/'logs/single_seed_analysis.log'
print('log_mtime',__import__('datetime').datetime.fromtimestamp(log.stat().st_mtime).isoformat());print(log.read_text(errors='replace')[-1800:])
PY
