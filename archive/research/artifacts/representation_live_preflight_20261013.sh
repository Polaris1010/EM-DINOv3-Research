set -euo pipefail
OUT=/opt/data/3_117/tanghr/codex/representation_diagnosis_live_20261013
test ! -e "$OUT"
mkdir -p "$OUT/scripts" "$OUT/sources" "$OUT/evidence" "$OUT/smoke" "$OUT/reports"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
curl -s -o /dev/null -w 'website_http=%{http_code}\n' --max-time 8 http://127.0.0.1:7860/
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import json,hashlib,shutil,inspect
from omegaconf import OmegaConf
r=Path('/opt/data/3_117/tanghr/codex/representation_diagnosis_live_20261013')
old=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
ext=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012')
stage=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002')
paths=[old/'configs/B_full.yaml',old/'scripts/train_B_full.py',old/'scripts/ssl_full_data.py',ext/'scripts/train_B_full.py',ext/'src/frozen_probe_150k.py',stage/'src/frozen_probe.py',stage/'src/frozen_probe_control.py',ext/'src/frozen_probe_label1.py',Path('/opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/primus.py')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
ref=json.loads((stage/'references/paths.json').read_text());paths.append(Path(ref['stage2_root'])/'stage1_reference/B_full_resolved_config.yaml')
import dynamic_network_architectures.building_blocks.patch_encode_decode as decoder_dep
paths.append(Path(inspect.getfile(decoder_dep)))
files=[]
for i,p in enumerate(paths):
 if p.is_file():
  dest=r/'sources'/(str(i)+'_'+p.name);shutil.copy2(p,dest)
  files.append({'path':str(p),'copy':str(dest),'sha256':sha(p),'bytes':p.stat().st_size})
 else:files.append({'path':str(p),'status':'UNVERIFIED missing'})
cfg=OmegaConf.load(old/'configs/B_full.yaml')
formal=json.loads((old/'data/pretrain_all_mitoverse.json').read_text())
shutil.copy2(old/'data/pretrain_all_mitoverse.json',r/'evidence/formal_pretrain_manifest.json')
inventory=[]
for root in [old/'runs/B_full/formal_attempt1/checkpoints',ext/'runs/B_full/formal_attempt1/checkpoints']:
 for p in sorted(root.glob('step*/teacher_backbone.pth')):inventory.append({'path':str(p),'step':int(p.parent.name.replace('step','')),'bytes':p.stat().st_size})
metrics=[]
for ds in ['Dataset111_CMCC','Dataset131_CMCT']:
 for kind,parent,name in [('P0',stage/'branch_E/control_20261011/runs','official_natural_'),('B100k',stage/'branch_E/runs','B_full_'),('B150k',ext/'evaluation','B_full150k_')]:
  folder=parent/(name+ds);p=folder/'metrics.json';d=json.loads(p.read_text())
  shutil.copy2(p,r/'evidence'/(kind+'_'+ds+'_metrics.json'))
  metrics.append({'id':kind,'dataset':ds,'path':str(p),'sha256':sha(p),'status':d['status'],'encoder':d['encoder'],'steps':d['steps'],'seed':d['seed'],'selected_step':d['datasets'][ds]['selected_step'],'macro_dice':d['datasets'][ds]['macro_dice'],'split_sha256':d['split_sha256'],'decoder_initial_sha256':d.get('decoder_initial_sha256','reconstructed audit required')})
logs=[]
def gram_scan(obj,prefix=''):
 found=[]
 if isinstance(obj,dict):
  for k,v in obj.items():
   name=prefix+'.'+k
   if 'gram' in k.lower():found.append((name,v))
   found.extend(gram_scan(v,name))
 return found
for root in [old/'runs/B_full/formal_attempt1',ext/'runs/B_full/formal_attempt1']:
 for p in sorted(root.glob('steps_rank*.jsonl')):
  n=0;nonzero=[];keys=set();first=None;last=None;loss_keys=set()
  with p.open() as f:
   for line in f:
    d=json.loads(line);n+=1;first=d if first is None else first;last=d
    loss_keys.update(d.get('loss_components',{}))
    for key,v in gram_scan(d):
     keys.add(key)
     if isinstance(v,(int,float)) and not isinstance(v,bool) and v!=0:nonzero.append({'step':d['step'],'key':key,'value':v})
  logs.append({'path':str(p),'sha256':sha(p),'line_count':n,'first_step':first['step'],'last_step':last['step'],'gram_fields':sorted(keys),'nonzero_gram_entries':nonzero,'loss_component_keys':sorted(loss_keys)})
result={'status':'live_collected','sources':files,'references':ref,'config':OmegaConf.to_container(cfg,resolve=True),'checkpoint_inventory':inventory,'historical_probes':metrics,'logs':logs,'formal_manifest_sha256':sha(old/'data/pretrain_all_mitoverse.json'),'manifest_top_keys':list(formal) if isinstance(formal,dict) else 'list','manifest_example':str(formal)[:2300],'audit_scope':[str(old),str(stage),str(ext)]}
(r/'evidence/live_evidence.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
PY
