import hashlib,json,sys,time
from pathlib import Path
import numpy as np,zarr

root=Path('/opt/data/3_117/tanghr')
r=root/'codex/mitoverse_round6_20260917_044606'
r4=root/'codex/mitoverse_round4_20260909_103239'
split=json.loads((r4/'split_mitoverse_probe_v1.json').read_text())
protocol=json.loads((r/'mitonet_comparison_protocol.json').read_text())
pp=json.loads((r/'exploratory_baseline_protocol.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
weights={}
for name,p,expected in [('P1_encoder',root/'DINOv3/med_checkpoints_continual_v2/eval/training_124999/teacher_checkpoint.pth',pp['encoder_sha256']),('MitoNet',root/'codex/mitotopo_20260908_round2/official/MitoNet_v1.pth',protocol['weight_sha256'])]:
 actual=sha(p);assert actual==expected;weights[name]={'path':str(p),'sha256':actual}
jobs=[]
for shard in (0,1):jobs.append(('MitoNet',None,r/'jobs'/f'mitonet_M0_shard{shard}_full','result.json'))
for seed in range(3):
 head=r/'jobs'/f'masked_P1_seed{seed}_10000/head_best.pth'
 assert sha(r4/f'shared_head_init_seed{seed}.pth')==pp['head_hashes'][str(seed)]
 actual=sha(head)
 weights[f'P1_head{seed}']={'path':str(head),'sha256':actual}
 for suffix in ('validation','legacy'):jobs.append(('P1',seed,r/'jobs'/f'masked_P1_seed{seed}_{suffix}','metrics.json'))
rows=[];spot=[];job_evidence=[]
for model,seed,job,file in jobs:
 d=json.loads((job/file).read_text())
 assert d.get('full',True) and d.get('status','complete')=='complete'
 if model=='P1' and 'head_checkpoint_sha256' in d:assert d['head_checkpoint_sha256']==weights[f'P1_head{seed}']['sha256']
 job_evidence.append({'model':model,'seed':seed,'job':str(job),'metrics_sha256':sha(job/file),'selected_step':d.get('selected_step'),'volumes':len(d['volumes'])})
 for vid,v in d['volumes'].items():
  counts={k:int(v[k]) for k in ('tp','fp','fn','tn')};tp,fp,fn,tn=[counts[k] for k in ('tp','fp','fn','tn')]
  dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1
  assert abs(dice-v['dice'])<1e-10
  src=split['volumes'][vid];sp='validation' if vid in split['validation'] else 'legacy_dev_test'
  group=zarr.open_group(str(Path(src['image_path']).parent),mode='r');raw=group['img'];label=group['mito'];pred=zarr.open_group(str(job/'predictions'/f'{vid}.zarr'),mode='r')
  assert pred['binary'].shape==raw.shape==label.shape
  assert v['slices']==raw.shape[0]
  if 'mask' not in group:assert sum(counts.values())==int(np.prod(raw.shape))
  elif 'valid_voxels' in v:assert sum(counts.values())==v['valid_voxels']
  z=raw.shape[0]//2;y=np.asarray(label[z])>0;p=np.asarray(pred['binary'][z])>0;valid=np.asarray(group['mask'][z])!=0 if 'mask' in group else np.ones(y.shape,bool)
  a=p[valid];b=y[valid];sc=[int((a&b).sum()),int((a&~b).sum()),int((~a&b).sum()),int((~a&~b).sum())]
  assert sum(sc)==int(valid.sum())
  # Float16 probabilities can round values across 0.5. Stored binary is authoritative.
  prob=np.asarray(pred['probability'][z]);mismatch=(p!=(prob>=.5))&valid
  far=int((mismatch & (np.abs(prob.astype(np.float32)-.5)>.001)).sum());assert far==0
  spot.append({'model':model,'seed':seed,'volume_id':vid,'z':z,'tp':sc[0],'fp':sc[1],'fn':sc[2],'tn':sc[3],'float16_threshold_rounding_pixels':int(mismatch.sum()),'far_threshold_mismatches':far})
  rows.append({'model':model,'seed':seed,'volume_id':vid,'source':v['source'],'split':sp,**counts,'dice':dice,'precision':v['precision'],'recall':v['recall']})
 print(f'Checked {job.name}',file=sys.stderr,flush=True)
for model,seed in [('MitoNet',None)]+[('P1',s) for s in range(3)]:
 rr=[x for x in rows if x['model']==model and x['seed']==seed]
 assert len(rr)==24 and len({x['volume_id'] for x in rr})==24
for vid in {x['volume_id'] for x in rows}:
 rr=[x for x in rows if x['volume_id']==vid]
 assert len({(x['tp']+x['fn'],x['fp']+x['tn']) for x in rr})==1
print(json.dumps({'checked_at_unix':time.time(),'weights':weights,'jobs':job_evidence,'rows':rows,'center_slice_checks':spot,'mitonet_protocol':protocol,'p1_protocol':pp,'scope':'Full-volume stored confusion counts, same foreground/background totals; current weight hashes checked; one center slice of each volume/model checked against stored probability. No new GPU inference.'}))
