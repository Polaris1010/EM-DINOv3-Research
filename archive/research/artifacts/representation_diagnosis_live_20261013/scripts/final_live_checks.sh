set -euo pipefail
OUT=/opt/data/3_117/tanghr/codex/representation_diagnosis_live_20261013
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,shutil,tarfile,subprocess,urllib.request
import torch
r=Path('/opt/data/3_117/tanghr/codex/representation_diagnosis_live_20261013')
ext=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012')
ev=json.loads((r/'evidence/live_evidence.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
inventory=[]
for row in ev['checkpoint_inventory']:
 p=Path(row['path']);ck=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
 meta={k:v for k,v in ck.items() if k!='backbone'}
 assert meta['step']==row['step'] and meta['manifest_sha256']==ev['formal_manifest_sha256']
 inventory.append(dict(row,sha256=sha(p),metadata=meta));del ck
(r/'evidence/checkpoint_inventory_verified.json').write_text(json.dumps(inventory,indent=2))
copied=[]
for folder in [ext/'audit',ext/'evaluation/label1_only_recompute_20261012']:
 for p in sorted(folder.glob('*.json')):
  if p.stat().st_size<2<<20:
   q=r/'evidence'/('final_'+p.name);shutil.copy2(p,q);copied.append({'path':str(p),'sha256':sha(p),'copy':str(q)})
device=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=8) as response:status=response.status
assert status==200
(r/'evidence/final_server_state.json').write_text(json.dumps({'gpus':device,'compute_apps':apps,'website_http':status,'copied_sources':copied},indent=2))
with tarfile.open(r/'evidence_bundle_final.tar.gz','w:gz') as tar:
 for folder in ['sources','evidence','smoke']:tar.add(r/folder,arcname=folder)
print(json.dumps({'verified_checkpoint_steps':[x['step'] for x in inventory],'website_http':status,'gpus':device,'copied':len(copied)},indent=2))
PY
