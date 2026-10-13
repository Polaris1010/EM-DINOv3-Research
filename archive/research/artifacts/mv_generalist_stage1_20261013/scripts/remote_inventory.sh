set -e
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import json, hashlib, subprocess, urllib.request
root=Path('/opt/data/3_117/tanghr/MitoVerse_data')
for p in sorted(root.iterdir()):
 print('ROOT',p.name,p.stat().st_size)
catalog=json.loads((root/'catalog.json').read_text())
print('SOURCES', {s:sum(v.get('dataset_id')==s for v in catalog.values()) for s in sorted({v.get('dataset_id','?') for v in catalog.values()})})
for s in sorted({v.get('dataset_id','?') for v in catalog.values()}):
 vs=[(k,v) for k,v in catalog.items() if v.get('dataset_id')==s]
 print('CATALOG_EXAMPLE',s,json.dumps(vs[:2]))
for p in sorted((root/'splits').glob('*.json')):
 print('SPLIT',p.name,p.read_text())
print('GPU',subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True))
print('GPU_PROCS',subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True))
print('SITE',urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10).status)
PY
