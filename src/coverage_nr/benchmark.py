"""Fixed MB1 development diagnostic for fixed100k encoders/new matched heads."""
import argparse,collections,csv,gc,math,os,sys,time
import numpy as np
import torch
from common import R,OLD,MV2,read,atomic,sha
from downstream import setup,module

class Planes:
 def __init__(self,path,shape):
  import tifffile
  self.shape=tuple(shape);self.file=None
  try:self.array=tifffile.memmap(path,mode='r')
  except (ValueError,TypeError):self.array=None;self.file=tifffile.TiffFile(path);assert len(self.file.pages)==shape[0]
  if self.array is not None:
   if self.array.ndim==2:self.array=self.array[None]
   assert self.array.shape==self.shape
 def __getitem__(self,z):return self.array[z] if self.array is not None else self.file.pages[z].asarray()
 def close(self):
  if self.file:self.file.close()

def main(model):
 _,_,probe=setup();seed=20261014;net,spec=probe.build(model,seed)
 head=R/'downstream/runs'/model/f'seed{seed}/head_step040000.pth'
 assert sha(head)==read(head.with_suffix('.json'))['sha256']
 allowed=[np.core.multiarray.scalar,np.dtype,type(np.dtype(np.float64))]
 with torch.serialization.safe_globals(allowed):saved=torch.load(head,map_location='cpu',weights_only=True,mmap=True)
 assert saved['step']==40000 and saved['metadata']['encoder_sha256']==spec['sha256']
 missing,extra=net.load_state_dict(saved['head'],strict=False);assert not extra and all(k.startswith('dino_encoder.') for k in missing)
 net.requires_grad_(False).eval();del saved;gc.collect()
 os.environ['MV2_GPU_EDT']='0';import validation as evaluator
 import tifffile
 reference=R.parent/'mb1_official_benchmark_20261015';manifest=read(reference/'MB1_benchmark_manifest.json');out=R/'benchmark'/model;out.mkdir(parents=True,exist_ok=True)
 images=[];started=time.time()
 for row in manifest['entries']:
  im=Planes(row['image_path'],row['shape_zyx']);truth=Planes(row['label_path'],row['shape_zyx'])
  path=out/(row['id']+'_prediction.npy');pred=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=row['shape_zyx'])
  for z in range(row['planes']):
   a=probe.sliding(net,im[z],tile_batch_size=16);pred[z]=a;b=np.asarray(truth[z])>0;c=probe.confusion(a,b);m=probe.overlap(c);tp,fp,fn,tn=c
   spacing=row['spacing_yx_nm']
   if spacing is None:hd=ns=None
   elif not tp+fn:hd=0. if not tp+fp else math.inf;ns=1. if not tp+fp else 0.
   elif not tp+fp:hd=math.inf;ns=0.
   else:hd,ns=evaluator.distances(a,b,np.ones(a.shape,bool),spacing)
   images.append({'model':model,'seed':seed,'dataset_id':row['dataset_id'],'dataset':row['dataset'],'image_id':row.get('image_id',''),'z':z,**m,'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,'hd95_nm':hd,'nsd_16nm':ns,'predicted_foreground_fraction':(tp+fp)/(tp+fp+fn+tn)})
   if z%50==0:atomic(out/'progress.json',{'dataset':row['dataset'],'z':z,'planes':row['planes'],'seconds':time.time()-started})
  pred.flush();im.close();truth.close();del pred,im,truth
 datasets=[]
 for bid in range(1,8):
  vv=[v for v in images if v['dataset_id']==bid];total={k:sum(v[k] for v in vv) for k in ['tp','fp','fn','tn']};tp,fp,fn,tn=(total[k] for k in ['tp','fp','fn','tn'])
  d={'model':model,'seed':seed,'dataset_id':bid,'dataset':vv[0]['dataset'],'planes':len(vv),**total,'volumetric_semantic_iou':tp/(tp+fp+fn) if bid<7 and tp+fp+fn else None}
  for key in ['dice','iou','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']:
   values=[v[key] for v in vv if v[key] is not None];d[key]=sum(values)/len(values) if values else None
  datasets.append(d)
 assert len(images)==2595
 atomic(out/'metrics.json',{'status':'complete','model':model,'images':images,'datasets':datasets,'encoder_sha256':spec['sha256'],'head_sha256':sha(head),'already_development_exposed':True,'hyperparameter_selection':False,'tta':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['Coverage-NR'],required=True);main(p.parse_args().model)
