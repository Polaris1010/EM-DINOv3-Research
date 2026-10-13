set -euo pipefail
TASK=/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013
systemctl show mv-dev-v2-notta-20261013.service --property=ActiveState,SubState,MainPID
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$TASK" <<'PY'
import sys,json,collections,pathlib,urllib.request,inspect
r=pathlib.Path(sys.argv[1])
for n in ['STATUS.json','MV_dev_validation_manifest.json','evidence/cache_completion.json']:
 p=r/n
 if not p.exists():continue
 j=json.loads(p.read_text())
 if n=='MV_dev_validation_manifest.json':
  print('MANIFEST',json.dumps({'tta':j['tta'],'entries':len(j['entries']),'sources':dict(collections.Counter(e['source_id'] for e in j['entries'])),'reuse':j['reuse_existing_MV1_dev_results']},ensure_ascii=False))
 else:print(n,json.dumps(j,ensure_ascii=False))
for model in ['P0','B100k','B150k']:
 ps=list((r/'runs'/model).glob('seed*/*_overlap.json'))
 origins=collections.Counter(json.loads(p.read_text())['result_origin'] for p in ps)
 print('ROWS',model,len(ps),dict(origins))
 print('FULL_METRIC_ROWS',model,len(list((r/'runs'/model).glob('seed*/*_metrics.json'))))
for p in sorted((r/'logs').glob('*.log')):
 print('LOG',p.name,'\n'+'\n'.join(p.read_text(errors='replace').splitlines()[-6:]))
with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=8) as f:print('WEBSITE_HTTP',f.status)
from monai.metrics.utils import get_mask_edges
print('MASK_EDGES_SIGNATURE',inspect.signature(get_mask_edges))
PY
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader
free -h
