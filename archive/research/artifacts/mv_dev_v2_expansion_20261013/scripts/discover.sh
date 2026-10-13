set -euo pipefail
TASK=/opt/data/3_117/tanghr/codex/mv_dev_v2_expansion_20261013
mkdir -p "$TASK"/scripts "$TASK"/src "$TASK"/evidence "$TASK"/sources "$TASK"/logs "$TASK"/runs
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import urllib.request,json,hashlib,subprocess
r=Path('/opt/data/3_117/tanghr/codex/mv_dev_v2_expansion_20261013')
urls={'mitoem_hf_tree':'https://huggingface.co/api/datasets/pytc/MitoEM2.0/tree/main?recursive=true&limit=1000',
 'mitoem_zenodo':'https://zenodo.org/api/records/20417683',
 'mitoverse_code_tree':'https://api.github.com/repos/PytorchConnectomics/mitoverse/git/trees/df884f6bf353b3cd6b4b6a5ee3620daabd8a79be?recursive=1',
 'mitoem_card':'https://huggingface.co/datasets/pytc/MitoEM2.0/raw/main/README.md'}
for name,url in urls.items():
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Codex-MVDev-split-audit'})
  with urllib.request.urlopen(req,timeout=45) as f:b=f.read()
  (r/'sources'/(name+'.txt')).write_bytes(b)
  print('FETCH',name,len(b),hashlib.sha256(b).hexdigest())
  if name=='mitoem_hf_tree':print('HF_PATHS',[v['path'] for v in json.loads(b) if 'split' in v['path'].lower() or v['type']=='file'][:50])
  if name=='mitoem_zenodo':print('ZENODO_FILES',[{'key':v['key'],'size':v['size'],'url':v['links']['self']} for v in json.loads(b)['files']])
  if name=='mitoverse_code_tree':print('CODE_PATHS',[v['path'] for v in json.loads(b)['tree'] if 'ingest' in v['path'] or 'han' in v['path'] or 'cellmap' in v['path']])
 except Exception as e:print('FETCH_FAILED',name,type(e).__name__,str(e))
print('TASK_PROCESSES',subprocess.check_output(['pgrep','-af','mv2_negative_transfer|mv_dev_v2_expansion'],text=True))
PY
