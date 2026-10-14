"""Official Semantic Only, capturing its probabilities from the same forward."""
import argparse,hashlib,time
import numpy as np
from shared import R,M4,MODEL,read,sha,atomic,rows,image_array,gpu_guard,config,output
from official import build

def inference(ids):
 import torch
 from empanada.inference.postprocess import find_instance_center
 gpu=gpu_guard();c=config();engine=build()
 params=read(M4/'MV4_protocol.json')['engine_parameters']
 engine.update_params(**{k:v for k,v in params.items() if k not in ['use_gpu','use_quantized'] and k!='semantic_only'},semantic_only=True)
 assert engine.engine.thing_list==[] and engine.engine.confidence_thr==0.5
 original=engine.engine.infer;captured={}
 def capture(image,render_steps=2):
  result=original(image,render_steps)
  assert result['sem'].shape[0:2]==(1,1)
  captured['prob']=result['sem'][0,0].detach().cpu().numpy().astype(np.float32,copy=True)
  captured['centers']=int(find_instance_center(result['ctr_hmp'],engine.engine.nms_threshold,engine.engine.nms_kernel).shape[0])
  return result
 engine.engine.infer=capture
 out=output();out.mkdir(exist_ok=True,parents=True)
 selected=[r for r in rows() if r['id'] in ids];assert len(selected)==len(ids)
 for row in selected:
  region=row['id'];receipt=out/(region+'_inference.json')
  if receipt.exists():assert read(receipt)['protocol_sha256']==sha(R/'MV4B_protocol.json');continue
  im,meta=image_array(row);shape=im.shape;started=time.time();h=hashlib.sha256()
  prob=np.lib.format.open_memmap(out/(region+'_probability.npy'),mode='w+',dtype=np.float32,shape=shape)
  pred=np.lib.format.open_memmap(out/(region+'_prediction.npy'),mode='w+',dtype=np.uint8,shape=shape)
  raw=np.lib.format.open_memmap(out/(region+'_raw_threshold.npy'),mode='w+',dtype=np.uint8,shape=shape)
  centers=[];small=[]
  for z in range(shape[0]):
   plane=np.asarray(im[z]);h.update(plane.tobytes(order='C'));seg=engine.infer(plane)>0
   p=captured.pop('prob')[:shape[1],:shape[2]];assert p.shape==plane.shape and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
   hard=p>=0.5
   # Official Semantic Only treats class1 as stuff; stuff_area64 applies to total per-plane class area.
   expected=hard if int(hard.sum())>=engine.engine.stuff_area else np.zeros_like(hard)
   assert np.array_equal(seg,expected),'Captured probability and official Semantic Only disagree'
   prob[z]=p;raw[z]=hard;pred[z]=seg;centers.append(captured.pop('centers'))
   if 0<int(hard.sum())<engine.engine.stuff_area:small.append(z)
   if z%20==0:
    atomic(out/(region+'_progress.json'),{'region_id':region,'z_completed':z+1,'z_total':shape[0],'physical_gpu':gpu})
    print('SEMANTIC_INFER',region,z+1,shape[0],gpu,flush=True)
  prob.flush();pred.flush();raw.flush();del prob,pred,raw
  atomic(receipt,{'status':'complete','stage':'MV-4B','model':MODEL,'region_id':region,'source_id':row['source_id'],
   'shape_zyx':list(shape),'native_region_zyx':row['region_zyx'],'input_cache':meta['paths']['image'],'input_image_sha256':h.hexdigest(),
   'protocol_sha256':sha(R/'MV4B_protocol.json'),'manifest_sha256':c['manifest_sha256'],'mv4_protocol_sha256':c['mv4_protocol_sha256'],
   'checkpoint_sha256':read(M4/'assets/official_assets.json')['files']['MitoNet_v1.pth']['sha256'],
   'physical_gpu':gpu,'gt_used_for_inference':False,'all_native_z_inferred':True,'tta':False,
   'semantic_only':True,'threshold':0.5,'probability_dtype':'float32','probability_source':'official RenderEngine.infer sem = logits_to_prob(sem_logits), captured during same official Engine2d forward before hardening',
   'official_semantic_output_equals_probability_threshold_after_stuff_area':True,'stuff_area':engine.engine.stuff_area,
   'small_foreground_planes_filtered_by_official_stuff_area':small,'detected_centers_per_plane':centers,'seconds':time.time()-started})
  print('SEMANTIC_REGION_COMPLETE',region,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--regions',nargs='+',required=True);a=p.parse_args();inference(a.regions)
