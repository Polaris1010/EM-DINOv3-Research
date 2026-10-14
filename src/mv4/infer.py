"""Official MitoNet batch-2D inference; predictions never depend on GT."""
import argparse,time
from pathlib import Path
import numpy as np
from common import R,MODEL,rows,image_array,read,atomic,config,sha,gpu_guard
from official import build

def infer(ids,smoke=False):
 gpu=gpu_guard();engine=build();c=config();t0=time.time()
 selected=[r for r in rows() if r['id'] in ids];assert len(selected)==len(ids)
 out=R/('smoke' if smoke else 'runs/Official-MitoNet-v1');out.mkdir(exist_ok=True,parents=True)
 for row in selected:
  receipt=out/(row['id']+'_inference.json')
  if receipt.exists():
   j=read(receipt);assert j['protocol_sha256']==sha(R/'MV4_protocol.json') and j['manifest_sha256']==c['manifest_sha256'];continue
  im,meta=image_array(row);zz=[0] if smoke else range(im.shape[0]);shape=(1,*im.shape[1:]) if smoke else im.shape
  pan=np.lib.format.open_memmap(out/(row['id']+'_panoptic.npy'),mode='w+',dtype=np.uint32,shape=shape)
  pred=np.lib.format.open_memmap(out/(row['id']+'_prediction.npy'),mode='w+',dtype=np.uint8,shape=shape)
  instances=np.lib.format.open_memmap(out/(row['id']+'_instances.npy'),mode='w+',dtype=np.uint32,shape=shape)
  counts=[];next_id=0;started=time.time()
  for k,z in enumerate(zz):
   seg=engine.infer(np.asarray(im[z]));assert seg.shape==im[z].shape and seg.min()>=0
   labels=np.unique(seg);labels=labels[labels>0]
   assert next_id+len(labels)<np.iinfo(np.uint32).max
   unique=np.zeros(seg.shape,np.uint32);positive=seg>0
   unique[positive]=np.searchsorted(labels,seg[positive]).astype(np.uint32)+next_id+1
   pan[k]=seg.astype(np.uint32);pred[k]=positive;instances[k]=unique
   counts.append({'z_local':int(z),'first_global_instance':next_id+1 if len(labels) else None,'instance_count':int(len(labels))});next_id+=len(labels)
   if k%10==0:
    atomic(out/(row['id']+'_progress.json'),{'region_id':row['id'],'z_completed':k+1,'z_total':len(zz),'seconds':time.time()-started,'physical_gpu':gpu})
    print('MITONET_INFER',row['id'],k+1,len(zz),'GPU',gpu,flush=True)
  pan.flush();pred.flush();instances.flush();del pan,pred,instances
  atomic(receipt,{'status':'complete','model':MODEL,'region_id':row['id'],'source_id':row['source_id'],
   'shape_zyx':list(shape),'original_region_zyx':row['region_zyx'],'spacing_zyx_nm':row['spacing_zyx_nm'],
   'spacing_status':row['spacing_status'],'manifest_sha256':c['manifest_sha256'],'protocol_sha256':sha(R/'MV4_protocol.json'),
   'checkpoint_sha256':read(R/'assets/official_assets.json')['files']['MitoNet_v1.pth']['sha256'],
   'physical_gpu':gpu,'tta':False,'gt_used_for_inference':False,'all_native_z_inferred':not smoke,
   'instance_interpretation':'official panoptic per-plane IDs plus disjoint global per-plane IDs; no 3D tracking claim',
   'instance_count':next_id,'plane_instances':counts,'seconds':time.time()-started})
 print('INFERENCE_BATCH_DONE',len(selected),time.time()-t0,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--regions',nargs='+',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();infer(a.regions,a.smoke)
