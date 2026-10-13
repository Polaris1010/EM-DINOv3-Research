"""CPU MitoNet scale check on the exact existing CMCC preprocessed image grid."""
import os,sys,json,time
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,psutil
from evaluate import BASE,ROOT,ROUND2,unpack,metrics,confusion
from postprocess import get_panoptic_segmentation,factor_pad
os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(2)
def main():
 out=BASE/'mitonet_pregrid_cpu';out.mkdir(exist_ok=True)
 model=torch.jit.load(str(ROUND2/'official/MitoNet_v1.pth'),map_location='cpu').eval()
 ds='Dataset111_CMCC';pre=ROOT/'nnUNet_preprocessed'/ds/'nnUNetPlans_2d';cases=json.loads((pre.parent/'splits_final.json').read_text())[0]['val']
 report={'status':'running','dataset':ds,'protocol':'MitoNet official normalization and postprocessing; input reconstructed from existing nnUNet preprocessed image grid; binary instance masks restored by nearest-neighbor, semantic logits by bilinear; CPU float32; original-grid validation','cases':{}}
 for case in cases:
  a=np.load(pre/(case+'.npy'),mmap_mode='r');gt=unpack(ROOT/'nnUNet_raw'/ds/'labelsTr'/(case+'.nii.gz'),case+'_gt');shape=(gt.shape[2],gt.shape[1],gt.shape[0]);assert a.shape[1]==shape[0]
  c=np.zeros(4,dtype=np.int64);cs=c.copy();pred=np.lib.format.open_memmap(out/(case+'.npy'),mode='w+',dtype=np.uint8,shape=shape);t0=time.time()
  for z in range(shape[0]):
   if z%20==0:assert psutil.virtual_memory().available>64*1024**3
   x=torch.from_numpy(((np.array(a[0,z],dtype=np.float32)*.229+.485)-.57571)/.12765)[None,None]
   h,w=x.shape[-2:]
   with torch.inference_mode():
    q=model(factor_pad(x));sem=(q['sem_logits'].sigmoid()>.5).long();p,_=get_panoptic_segmentation(sem,q['ctr_hmp'],q['offsets'],[1],1000,64,0,.1,7)
    p=p.reshape(1,1,*p.shape[-2:])[...,:h,:w]
    y=(F.interpolate((p>0).float(),size=shape[1:],mode='nearest')[0,0]>0).numpy()
    ys=(F.interpolate(q['sem_logits'][...,:h,:w],size=shape[1:],mode='bilinear',align_corners=False)[0,0]>0).numpy()
   true=np.asarray(gt.dataobj[:,:,z]).T>0;c+=confusion(y,true);cs+=confusion(ys,true);pred[z]=y
   if z%50==0:print(case,z,metrics(c)['dice'],flush=True)
  pred.flush();report['cases'][case]={**metrics(c),'semantic_head_only':metrics(cs),'elapsed':time.time()-t0};(out/'metrics.json').write_text(json.dumps(report,indent=2))
 report['macro_dice']=float(np.mean([v['dice'] for v in report['cases'].values()]));report['status']='complete';(out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
