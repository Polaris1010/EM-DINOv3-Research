"""MV-3 inference and the existing MV2 physical-distance evaluator."""
import argparse,gc,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
from data import R,OLD,MV2,read,sha,atomic,manifest,open_arrays
sys.path[:0]=[str(OLD/'src'),str(MV2/'src')]
from mv_probe import build,sliding,confusion,overlap,state_hash,head_state
from train import gpu_guard

def infer(model,seed):
 gpu_guard();out=R/'runs'/model/f'seed{seed}';protocol=read(R/'MV3_protocol.json');path=out/'head_step040000.pth'
 assert read(out/'training_complete.json')['completed_steps']==40000
 checkpoint_hash=sha(path);saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 assert saved['metadata']['model']==model and saved['metadata']['seed']==seed and saved['step']==40000
 net,spec=build(protocol['models'][model],seed);missing,extra=net.load_state_dict(saved['head'],strict=False)
 assert not extra and all(k.startswith('dino_encoder.') for k in missing);assert state_hash(head_state(net))==state_hash(saved['head'])
 net.requires_grad_(False).eval();del saved;gc.collect();t0=time.time()
 rows=[r for r in manifest()['entries'] if r['supervised_role']=='dev']
 for row in sorted(rows,key=lambda r:np.prod([b-a for a,b in r['region_zyx']])):
  target=out/(row['id']+'_overlap.json')
  if target.exists():
   previous=read(target);assert previous['head_sha256']==checkpoint_hash and previous['manifest_sha256']==sha(R/'MV-Generalist-v1.json');continue
  im,gt,valid,j=open_arrays(row);path=out/(row['id']+'_prediction.npy')
  pred=np.lib.format.open_memmap(path,mode='w+',shape=im.shape,dtype=np.uint8);pred[:]=0;c=np.zeros(4,np.int64)
  for k,z in enumerate(j['valid_z']):
   p=sliding(net,im[z],tile_batch_size=protocol['inference_tile_batch_size']);pred[z]=p
   mask=valid[z].astype(bool) if valid is not None else np.ones(p.shape,bool);c+=confusion(p[mask],gt[z].astype(bool)[mask])
   if k%20==0:atomic(out/'progress.json',{'stage':'full_development_inference','model':model,'seed':seed,'region_id':row['id'],'z_completed':k+1,'z_total':len(j['valid_z']),'seconds':time.time()-t0})
  pred.flush();del pred
  result={'stage':'MV-3','model':model,'seed':seed,'region_id':row['id'],'volume_id':row['volume_id'],'source_id':row['source_id'],'dev_kind':row['dev_kind'],'acquisition_id':row['acquisition_id'],'region_zyx':row['region_zyx'],'spacing_zyx_nm':row['spacing_zyx_nm'],'spacing_status':row['spacing_status'],'valid_mask_policy':row['valid_mask_policy'],'head_sha256':checkpoint_hash,'encoder_sha256':spec['sha256'],'manifest_sha256':sha(R/'MV-Generalist-v1.json'),'protocol_sha256':sha(R/'MV3_protocol.json'),'tta':False,**overlap(c)}
  result.update(valid_voxels=int(c.sum()),gt_foreground_voxels=int(c[0]+c[2]),prediction_foreground_voxels=int(c[0]+c[1]),predicted_foreground_fraction=float((c[0]+c[1])/c.sum()))
  assert result['valid_voxels']==j['valid_voxels'];atomic(target,result);print('INFERRED',model,seed,row['id'],result['dice'],flush=True)
 atomic(out/'inference_complete.json',{'status':'complete','regions':len(rows),'sources':12,'manifest_sha256':sha(R/'MV-Generalist-v1.json'),'head_sha256':checkpoint_hash,'tta':False,'seconds':time.time()-t0})

def score(path):
 gpu_guard();os.environ['MV2_GPU_EDT']='1'
 # Imports the already verified global exact EDT + MONAI metric functions. No installed-package changes.
 import validation as evaluator
 import gpu_edt
 result=read(path);out=path.with_name(path.name.replace('_overlap.json','_metrics.json'))
 if out.exists():assert read(out)['head_sha256']==result['head_sha256'];return
 row=next(r for r in manifest()['entries'] if r['id']==result['region_id']);im,lab,valid,j=open_arrays(row)
 gt=lab.astype(bool);valid=valid.astype(bool) if valid is not None else np.ones(gt.shape,bool)
 pred=np.load(path.with_name(path.name.replace('_overlap.json','_prediction.npy')),mmap_mode='r')>0
 assert pred.shape==gt.shape==valid.shape
 t0=time.time()
 if not result['gt_foreground_voxels']:hd=0. if not result['prediction_foreground_voxels'] else math.inf;ns=1. if not result['prediction_foreground_voxels'] else 0.
 elif not result['prediction_foreground_voxels']:hd=math.inf;ns=0.
 else:hd,ns=evaluator.distances(pred,gt,valid,row['spacing_zyx_nm'])
 result.update(hd95_nm=hd,nsd_16nm=ns,metric_seconds=time.time()-t0,distance_evaluator_source_sha256=sha(MV2/'src/validation.py'),distance_backend='MV2_existing_exact_EDT_MONAI')
 atomic(out,result);gpu_edt.release();print('SCORED',result['model'],result['seed'],result['region_id'],hd,ns,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['infer','score']);ap.add_argument('--model',choices=['G-P0','G-B100']);ap.add_argument('--seed',type=int);ap.add_argument('--path',type=Path);a=ap.parse_args()
 if a.mode=='infer':infer(a.model,a.seed)
 else:score(a.path)
