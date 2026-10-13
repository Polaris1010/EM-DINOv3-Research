set -euo pipefail
MV1=/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013
PY=/opt/data/3_117/tanghr/envs/meddinov3/bin/python
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
free -h
df -h "$MV1"
"$PY" - <<'PY'
from pathlib import Path
import json,torch,importlib.util,urllib.request
r=Path('/opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013')
for model in ['P0','B100k']:
 p=r/'runs'/model/'seed20261013'/'head_last.pth'
 s=torch.load(p,map_location='cpu',weights_only=True)
 print('HEAD',model,{'keys':list(s),'step':s['step'],'files':[v.name for v in p.parent.glob('*.pth')]})
for name in ['empanada','empanada_napari','sklearn','zarr','matplotlib','scipy']:
 print('MODULE',name,importlib.util.find_spec(name) is not None)
p=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/data/pretrain_all_mitoverse.json')
s=json.loads(p.read_text());print('SSL_KEYS',list(s) if isinstance(s,dict) else 'list');print('SSL_ENTRY',str(s)[:2500])
for p in (r/'runs').glob('*/seed*/curve.json'):
 print('CURVE',p.parent.parent.name,p.parent.name,[{'step':v['step'],'loss':v['loss_median200'],'sources':v['source_means']} for v in json.loads(p.read_text())])
print('WEBSITE',urllib.request.urlopen('http://127.0.0.1:7860/',timeout=8).status)
PY
rg -n 'sampler|seed|sample|rng' /opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/src/stage1_ssl_full_data.py || true
