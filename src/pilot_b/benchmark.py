"""Fixed seven-group MB1 inference; semantic AP/remote FP on muscle only."""
import argparse,gc,inspect,os,sys,time,math,collections
import numpy as np
import torch
from common import R,B,PA,A2,PARENT,LATE,MODEL,MV2,read,atomic,sha,module
from downstream import setup
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
 _,_,probe=setup();net,spec=probe.build(model,20261014)
 root=R if model==MODEL else PARENT if model=='D2-30k' else LATE
 head=root/'downstream/runs'/(MODEL if model==MODEL else 'D2')/'seed20261014/head_step040000.pth'
 assert sha(head)==read(head.with_suffix('.json'))['sha256']
 allowed=[np.core.multiarray.scalar,np.dtype,type(np.dtype(np.float64))]
 with torch.serialization.safe_globals(allowed):saved=torch.load(head,map_location='cpu',weights_only=True,mmap=True)
 assert saved['step']==40000 and saved['metadata']['encoder_sha256']==spec['sha256']
 missing,extra=net.load_state_dict(saved['head'],strict=False);assert not extra and all(k.startswith('dino_encoder.') for k in missing)
 net.requires_grad_(False).eval();del saved;gc.collect()
 # Identical sliding implementation; only expose its pre-threshold return value.
 text=inspect.getsource(probe.sliding);assert text.count('return acc/weight>.5')==1
 text=text.replace('def sliding(', 'def probability_sliding(').replace('return acc/weight>.5','return acc/weight')
 namespace={};exec(compile(text,'identical_sliding_with_probability_return','exec'),probe.__dict__,namespace);probability=namespace['probability_sliding']
 os.environ.setdefault('PILOT_A2_ROOT',str(A2));metrics=module('pilot_b_pixel_AP',A2/'code/common.py')
 import validation as evaluator
 import gpu_edt
 manifest=read(B/'mb1_official_benchmark_20261015/MB1_benchmark_manifest.json');out=R/'benchmark'/model;out.mkdir(parents=True,exist_ok=True)
 rows=[];started=time.time();old_differences=0
 for row in manifest['entries']:
  muscle=row['dataset']=='glycolytic_muscle'
  if model!=MODEL and not muscle:continue
  im=Planes(row['image_path'],row['shape_zyx']);gt=Planes(row['label_path'],row['shape_zyx'])
  pred=np.lib.format.open_memmap(out/(row['id']+'_prediction.npy'),mode='w+',shape=tuple(row['shape_zyx']),dtype=np.uint8)
  probabilities=np.lib.format.open_memmap(out/(row['id']+'_probability.npy'),mode='w+',shape=tuple(row['shape_zyx']),dtype=np.float32) if muscle else None
  legacy=module('pilot_b_legacy_array_io',PA/'code/common.py')
  historical=legacy.npy(root/'benchmark/D2'/(row['id']+'_prediction.npy')) if model!=MODEL else None
  for z in range(row['planes']):
   if muscle:
    prob=probability(net,im[z],tile_batch_size=16);a=prob>.5;probabilities[z]=prob
    if z==0:assert np.array_equal(a,probe.sliding(net,im[z],tile_batch_size=16))
   else:a=probe.sliding(net,im[z],tile_batch_size=16)
   pred[z]=a;b=np.asarray(gt[z])>0;c=probe.confusion(a,b);m=probe.overlap(c);tp,fp,fn,tn=c;spacing=row['spacing_yx_nm']
   if historical is not None:old_differences+=int(np.count_nonzero(a!=historical[z]))
   if model!=MODEL:hd=ns=None
   elif spacing is None:hd=ns=None
   elif not tp+fn:hd=0. if not tp+fp else math.inf;ns=1. if not tp+fp else 0.
   elif not tp+fp:hd=math.inf;ns=0.
   else:hd,ns=evaluator.distances(a,b,np.ones(a.shape,bool),spacing)
   ap=auc=None;nfar=ffp=None
   if muscle:
    score=torch.as_tensor(prob,device='cuda');target=torch.as_tensor(b,device='cuda');rank=metrics.Ranked(score.flatten(),target.flatten(),torch.zeros(score.numel(),device='cuda',dtype=torch.int64));auc,ap,_=rank.calculate()
    distance=gpu_edt.edt_array(~b,spacing) if b.any() else np.full(b.shape,np.inf);far=(distance>180)&~b;nfar=int(far.sum());ffp=int((a&far).sum());del rank,score,target,distance
   rows.append({'model':model,'seed':20261014,'dataset_id':row['dataset_id'],'dataset':row['dataset'],'image_id':row.get('image_id',''),'z':z,**m,'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,'hd95_nm':hd,'nsd_16nm':ns,'predicted_foreground_fraction':(tp+fp)/(tp+fp+fn+tn),'pixel_AUROC':auc,'pixel_AP':ap,'far_background_pixels':nfar,'far_FP_pixels':ffp,'far_FP_rate':ffp/nfar if nfar else None})
   if z%25==0:atomic(out/'progress.json',{'dataset':row['dataset'],'z':z,'planes':row['planes'],'seconds':time.time()-started})
  pred.flush()
  if probabilities is not None:probabilities.flush()
  im.close();gt.close();del pred,probabilities,im,gt,historical
 datasets=[]
 for bid in sorted({x['dataset_id'] for x in rows}):
  a=[x for x in rows if x['dataset_id']==bid];total={k:sum(x[k] for x in a) for k in ['tp','fp','fn','tn']};tp,fp,fn,tn=(total[k] for k in ['tp','fp','fn','tn'])
  d={'model':model,'dataset_id':bid,'dataset':a[0]['dataset'],'planes':len(a),**total,'volumetric_semantic_iou':tp/(tp+fp+fn) if bid<7 and tp+fp+fn else None}
  for key in ['dice','iou','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction','pixel_AUROC','pixel_AP']:
   values=[x[key] for x in a if x[key] is not None];d[key]=sum(values)/len(values) if values else None
  far=[x for x in a if x['far_background_pixels'] is not None];den=sum(x['far_background_pixels'] for x in far)
  d['far_FP_rate']=sum(x['far_FP_pixels'] for x in far)/den if den else None;datasets.append(d)
 if model==MODEL:assert len(rows)==2595 and len(datasets)==7
 else:assert old_differences==0,'Historical endpoint masks differ; do not silently alter comparator'
 atomic(out/'metrics.json',{'status':'complete','model':model,'images':rows,'datasets':datasets,'encoder_sha256':spec['sha256'],'head_sha256':sha(head),'seconds':time.time()-started,'tta':False,'old_mask_disagreement_pixels':old_differences,'AP_scope':'muscle mean per-plane semantic pixel AP, empty GT excluded; no threshold selection','probability_path_scope':'muscle only; float32 same sliding mean before threshold','coefficient_selection':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=[MODEL,'D2-30k','D2-100k'],default=MODEL);a=p.parse_args()
 from gpu_scoring import accelerated
 with accelerated('MB1_'+a.model):main(a.model)
