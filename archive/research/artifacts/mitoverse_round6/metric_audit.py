"""Serial I/O audit of saved full-volume predictions. Historical results stay read-only."""
import os,sys,json,csv,time,hashlib
from pathlib import Path
from settings import *
sys.path.insert(0,str(ROOT/'codex/mitotopo_20260908_round2/deps'))
import numpy as np,zarr
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,default=str));t.replace(p)
def table(name,rows):
 if not rows:return
 with (BASE/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(c):
 tp,fp,fn,tn=map(int,c);n=int(sum(c))
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,precision=tp/(tp+fp) if tp+fp else (0. if fn else 1.),recall=tp/(tp+fn) if tp+fn else 1.,true_foreground_fraction=(tp+fn)/n,predicted_foreground_fraction=(tp+fp)/n)
def main():
 out=Path(os.environ['RUN_OUTPUT']);split=json.loads((R4/'split_mitoverse_probe_v1.json').read_text());jobs=[];rows=[];index=[];start=time.time()
 for model in ['P1','G','C','X1']:
  for seed in range(3):
   p=R4/f'jobs/evaluate_P1_seed{seed}_N2' if model=='P1' else R5/'jobs'/(f'v4_b16_X1_seed{seed}_eval' if model=='X1' else f'v4_b16_{model}_30000_head{seed}_eval')
   meta=json.loads((p/'metrics.json').read_text());assert meta['full'];jobs.append((model,seed,p,meta))
 save(out/'executed_source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')})
 from PIL import Image
 examples=[]
 for source in sorted({split['volumes'][v]['source'] for v in split['validation']}):
  vid=next(v for v in split['validation'] if split['volumes'][v]['source']==source);r=split['volumes'][vid];a=zarr.open_array(r['image_path'],mode='r');b=zarr.open_array(r['label_path'],mode='r');assert a.shape==b.shape
  d,h,w=a.shape;z=d//2;y=max(0,(h-256)//2);x=max(0,(w-256)//2);im=np.asarray(a[z,y:y+256,x:x+256]);lab=np.asarray(b[z,y:y+256,x:x+256])>0
  rgb=np.repeat(im[:,:,None],3,axis=2);edge=lab ^ __import__('scipy.ndimage',fromlist=['binary_erosion']).binary_erosion(lab);rgb[edge]=[0,255,0];Image.fromarray(rgb).save(BASE/f'validation_fixed_{source}.png')
  norm=(np.repeat(im[None].astype(np.float32)/255,3,axis=0)-np.array([.485,.456,.406])[:,None,None])/np.array([.229,.224,.225])[:,None,None]
  examples.append(dict(volume_id=vid,source=source,split='validation',z=z,y=y,x=x,axes='ZYX',shape=list(a.shape),voxel_nm_zyx=r['voxel_nm_z_y_x'],image_dtype=str(a.dtype),n2_range=[float(norm.min()),float(norm.max())],foreground_pixels=int(lab.sum()),same_crop_coordinates=True,interpolation='none'))
 save(BASE/'validation_patch_audit.json',examples)
 vids=split['validation']+split['test1']+split['test2']+split['test3']
 prior=BASE/'metric_audit.csv';completed=set()
 if prior.exists():
  with prior.open() as f:
   for r in csv.DictReader(f):
    r['head_seed']=int(r['head_seed'])
    for key in ['tp','fp','fn','tn']:r[key]=int(r[key])
    for key in ['dice','precision','recall','true_foreground_fraction','predicted_foreground_fraction']:r[key]=float(r[key])
    r['historical_confusion_exact']=r['historical_confusion_exact']=='True'
    model,seed,path,meta=next(j for j in jobs if j[0]==r['model'] and j[1]==r['head_seed'])
    assert all(r[k]==meta['volumes'][r['volume_id']][k] for k in ['tp','fp','fn','tn'])
    rows.append(r)
  completed={r['volume_id'] for r in rows}
  assert all(sum(r['volume_id']==vid for r in rows)==12 for vid in completed)
  assert completed==set(vids[:len(completed)]),'Only a completed volume prefix can be reused'
  index=json.loads((BASE/'prediction_index.json').read_text());assert len(index)==len(rows)
 save(out/'recount_resume.json',dict(completed_volumes_reused=len(completed),row_count=len(rows),historical_confusions_reverified=True))
 for vid in vids:
  if vid in completed:continue
  r=split['volumes'][vid];label=zarr.open_array(r['label_path'],mode='r');mask_path=Path(r['label_path']).parent/'mask'
  assert not mask_path.exists(),'Held-out mask requires explicit polarity; do not guess'
  arrays=[];counts=np.zeros((len(jobs),4),dtype=np.int64)
  for model,seed,p,meta in jobs:
   store=p/'predictions'/f'{vid}.zarr';g=zarr.open_group(str(store),mode='r');a=g['binary'];assert a.shape==label.shape
   assert dict(g.attrs).get('threshold')==.5
   arrays.append(a);index.append(dict(model=model,head_seed=seed,volume_id=vid,path=str(store),shape=list(a.shape),valid_region='whole volume; no mask supplied',probability_dtype=str(g['probability'].dtype),binary_source='original float32 threshold, before float16 storage'))
  dz,dy,dx=label.chunks;D,H,W=label.shape
  for z in range(0,D,dz):
   for y in range(0,H,dy):
    for x in range(0,W,dx):
     sl=(slice(z,min(z+dz,D)),slice(y,min(y+dy,H)),slice(x,min(x+dx,W)));t=np.asarray(label[sl])>0
     for k,a in enumerate(arrays):
      p=np.asarray(a[sl]);assert np.all((p==0)|(p==1));p=p!=0
      tp=int(np.count_nonzero(p&t));fp=int(np.count_nonzero(p&~t));fn=int(np.count_nonzero(~p&t));counts[k]+=[tp,fp,fn,t.size-tp-fp-fn]
   save(out/'progress.json',dict(volume_id=vid,z_completed=min(z+dz,D),z_total=D,completed_volumes=len(rows)//len(jobs),seconds=time.time()-start))
  for k,(model,seed,p,meta) in enumerate(jobs):
   m=metrics(counts[k]);old=meta['volumes'][vid];assert sum(counts[k])==int(np.prod(label.shape))
   match=all(m[key]==old[key] for key in ['tp','fp','fn','tn']);assert match,f'Historical confusion mismatch {model} {seed} {vid}'
   rows.append(dict(model=model,head_seed=seed,volume_id=vid,source=r['source'],split='validation' if r['split']=='validation' else 'legacy_dev_test',**m,historical_confusion_exact=match))
  table('metric_audit.csv',rows);save(BASE/'prediction_index.json',index);print('AUDITED',vid,'seconds',time.time()-start,flush=True)
 sources=[]
 for model,seed,p,meta in jobs:
  for source in sorted({r['source'] for r in rows}):
   rr=[r for r in rows if r['model']==model and r['head_seed']==seed and r['source']==source]
   sources.append(dict(model=model,head_seed=seed,source=source,split=rr[0]['split'],volumes=len(rr),**{key:float(np.mean([r[key] for r in rr])) for key in ['dice','precision','recall','true_foreground_fraction','predicted_foreground_fraction']}))
 table('comparison_volume_seed.csv',rows);table('comparison_source_seed.csv',sources);paired=[]
 for other in ['G','C','X1']:
  for s in sources:
   if s['model']!=other:continue
   ref=next(r for r in sources if r['model']=='P1' and r['head_seed']==s['head_seed'] and r['source']==s['source'])
   paired.append(dict(comparison=other+'-P1',head_seed=s['head_seed'],source=s['source'],split=s['split'],dice_delta=s['dice']-ref['dice'],precision_delta=s['precision']-ref['precision'],recall_delta=s['recall']-ref['recall'],scope='historical models; not Round6 TM/TJ/T0'))
 table('paired_deltas.csv',paired)
 save(out/'result.json',dict(status='complete',volumes=len(vids),models=4,head_seeds=3,rows=len(rows),all_historical_confusions_exact=True,seconds=time.time()-start,aggregation='equal volumes within source, equal sources; validation separate from legacy_dev_test',float16_note='Use saved binary, not rethresholded float16 probabilities.'))
if __name__=='__main__':main()
