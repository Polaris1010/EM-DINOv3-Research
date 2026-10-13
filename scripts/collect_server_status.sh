set -u
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
from pathlib import Path
import json,datetime,subprocess,urllib.request
root=Path('/opt/data/3_117/tanghr/codex')
result={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'files':{},'services':{}}
for name in ['mv_generalist_stage1_20261013','mv_dev_v2_notta_20261013','mv_dev_v2_expansion_20261013']:
 for filename in ['STATUS.json','inference_all_complete.json','metrics_all_complete.json','STATUS_TTA_SUPERSEDED.json']:
  p=root/name/filename
  if p.is_file():result['files'][str(p)]=json.loads(p.read_text())
for service in ['mv-generalist-stage1-20261013.service','mv-dev-v2-notta-20261013.service','mv-dev-v2-expansion-20261013.service']:
 result['services'][service]=subprocess.run(['systemctl','show',service,'-p','ActiveState','-p','SubState','-p','MainPID','-p','Result'],capture_output=True,text=True).stdout
try:
 request=urllib.request.Request('http://127.0.0.1:7860/',method='HEAD')
 with urllib.request.urlopen(request,timeout=10) as response:result['website_http']=response.status
except Exception as error:result['website_error']=str(error)
result['gpu_read_only_query']=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(result))
PY
