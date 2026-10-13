"""Descriptive full-volume expansion of fixed MitoNet and old P1 checkpoints."""
import argparse,json,os,sys,time
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parent
ROOT=Path('/opt/data/3_117/tanghr');R4=ROOT/'codex/mitoverse_round4_20260909_103239';R6=ROOT/'codex/mitoverse_round6_20260917_044606'
sys.path.insert(0,str(R4))
import common as c
import train as old
import numpy as np,torch,zarr
from scipy.ndimage import gaussian_filter

class Infer:
 def __init__(self,kind,seed):
  self.kind=kind;self.bs=8 if kind=='P1' else 2;self.reductions=[]
  if kind=='P1':
   protocol=json.loads((R6/'exploratory_baseline_protocol.json').read_text());assert c.sha(c.PATHS['P1'])==protocol['encoder_sha256']
   head=R6/f'jobs/masked_P1_seed{seed}_10000/head_best.pth'
   ck=torch.load(head,map_location='cpu',weights_only=False)
   self.net=c.model('P1',seed);c.load_head(self.net,ck['head']);self.net.eval()
   self.identity={'encoder_sha256':c.sha(c.PATHS['P1']),'head_sha256':c.sha(head),'head_step':ck['step']}
  else:
   weight=ROOT/'codex/mitotopo_20260908_round2/official/MitoNet_v1.pth'
   protocol=json.loads((R6/'mitonet_comparison_protocol.json').read_text());assert c.sha(weight)==protocol['weight_sha256']
   self.net=torch.jit.load(str(weight),map_location='cpu').eval().cuda();self.identity={'weight_sha256':protocol['weight_sha256']}
 def forward(self,patches):
  if self.kind=='P1':
   x=torch.from_numpy(np.stack([c.normalize(p,'N2') for p in patches])).cuda()
   return c.forward(self.net,x).float().softmax(1)[:,1].cpu().numpy()
  x=torch.from_numpy(np.stack([(p.astype(np.float32)/255-.57571)/.12765 for p in patches])[:,None]).cuda()
  h,w=x.shape[-2:];py=(-h)%16;px=(-w)%16
  if py or px:x=torch.nn.functional.pad(x,(0,px,0,py))
  return self.net(x)['sem_logits'][:,0,:h,:w].float().cpu().numpy()
 @torch.inference_mode()
 def __call__(self,raw):
  h,w=raw.shape
  if self.kind=='MitoNet' and max(h,w)<=512:
   q=self.forward([raw])[0];return torch.from_numpy(q).sigmoid().numpy()
  size=256 if self.kind=='P1' else 512
  im=np.pad(raw,((0,max(0,size-h)),(0,max(0,size-w))),mode='reflect') if self.kind=='P1' else raw
  H,W=im.shape
  if self.kind=='P1':
   ys,xs=old.starts(H),old.starts(W)
   g=np.zeros((256,256),np.float32);g[128,128]=1;g=gaussian_filter(g,32);g/=g.max();g=np.maximum(g,1e-7)
  else:
   starts=lambda n:[0] if n<=512 else list(range(0,n-512,256))+[n-512]
   ys,xs=starts(H),starts(W)
   ax=torch.arange(512,device='cuda').float()-255.5;v=torch.exp(-.5*(ax/64)**2);g=(v[:,None]*v[None,:]).clamp_min(1e-7).cpu().numpy()
  coords=[(y,x) for y in ys for x in xs];pred=np.zeros((H,W),np.float32);den=np.zeros_like(pred);i=0
  while i<len(coords):
   chunk=coords[i:i+self.bs]
   try:q=self.forward([im[y:y+size,x:x+size] for y,x in chunk])
   except torch.cuda.OutOfMemoryError:
    if self.bs==1:raise
    self.reductions.append({'from':self.bs,'to':max(1,self.bs//2)});self.bs=max(1,self.bs//2);torch.cuda.empty_cache();continue
   assert np.isfinite(q).all()
   for (y,x),p in zip(chunk,q):
    ph,pw=p.shape;pred[y:y+ph,x:x+pw]+=p*g[:ph,:pw];den[y:y+ph,x:x+pw]+=g[:ph,:pw]
   i+=len(chunk)
  assert (den>0).all();q=(pred/den)[:h,:w]
  return q if self.kind=='P1' else torch.from_numpy(q).sigmoid().numpy()

@torch.inference_mode()
def reference_mito(net,raw):
 x=torch.from_numpy((raw.astype(np.float32)/255-.57571)/.12765)[None,None].cuda();h,w=raw.shape
 def padded(t):
  ph,pw=t.shape[-2:];return torch.nn.functional.pad(t,(0,(-pw)%16,0,(-ph)%16))
 if max(h,w)<=512:return net(padded(x))['sem_logits'].sigmoid()[0,0,:h,:w].cpu().numpy()
 starts=lambda n:[0] if n<=512 else list(range(0,n-512,256))+[n-512]
 accum=torch.zeros((1,1,h,w),device='cuda');den=torch.zeros_like(accum)
 ax=torch.arange(512,device='cuda').float()-255.5;g=torch.exp(-.5*(ax/64)**2);g=(g[:,None]*g[None,:]).clamp_min(1e-7)[None,None]
 for y in starts(h):
  for z in starts(w):
   patch=x[...,y:y+512,z:z+512];ph,pw=patch.shape[-2:];q=net(padded(patch))['sem_logits'][...,:ph,:pw];weight=g[...,:ph,:pw]
   accum[...,y:y+ph,z:z+pw]+=q*weight;den[...,y:y+ph,z:z+pw]+=weight
 return (accum/den).sigmoid()[0,0].cpu().numpy()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['P1','MitoNet'],required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 out=Path(os.environ['RUN_OUTPUT']);torch.set_num_threads(2);c.seed(a.seed)
 infer=Infer(a.model,a.seed);start=time.time();rows=json.loads((R/'data/expanded_eval_manifest.json').read_text())
 if a.smoke:
  rows=sorted(rows,key=lambda r:np.prod(r['shape'][1:]),reverse=True)[:1]
 result={};verification=[]
 for row in rows:
  g=zarr.open_group(row['local_path'],mode='r');raw=g['img'];lab=g['mito'];region=g['mask'] if 'mask' in g else None
  counts=np.zeros(4,np.int64);zs=[raw.shape[0]//2] if a.smoke else range(raw.shape[0]);valid_total=0
  for z in zs:
   image=np.asarray(raw[z]);truth=np.asarray(lab[z])>0;valid=np.asarray(region[z])!=0 if region is not None else np.ones(image.shape,bool)
   prob=infer(image);prediction=prob>=.5
   counts+=c.confusion(prediction[valid],truth[valid]);valid_total+=int(valid.sum())
   if a.smoke and a.model=='P1':
    original=old.slide(infer.net,image,'N2');assert np.max(np.abs(original-prob))<1e-5
    verification.append({'max_abs_probability_difference':float(np.max(np.abs(original-prob))),'binary_different_pixels':int(((original>=.5)!=prediction).sum()),'reference':'original P1 slide, bs8'})
   if a.smoke and a.model=='MitoNet':
    original=reference_mito(infer.net,image);difference=float(np.max(np.abs(original-prob)));assert difference<1e-5
    verification.append({'max_abs_probability_difference':difference,'binary_different_pixels':int(((original>=.5)!=prediction).sum()),'reference':'original fixed M0 tile order and GPU Gaussian logit fusion, bs1'})
   if z==raw.shape[0]//2:
    (out/'center_slices').mkdir(exist_ok=True)
    np.savez_compressed(out/'center_slices'/f"{row['volume_id']}.npz",image=image,truth=truth,valid=valid,prediction=prediction)
   if z%25==0:c.save(out/'progress.json',{'model':a.model,'seed':a.seed,'volume':row['volume_id'],'z':z,'depth':raw.shape[0],'completed_volumes':len(result),'total_volumes':len(rows),'batch':infer.bs,'seconds':time.time()-start})
  assert sum(counts)==valid_total
  result[row['volume_id']]={'source':row['original_dataset_id'],'split':row['split'],'interpretation':row['interpretation'],
   'mask_semantics':row['mask_semantics'],'slices':len(zs),'valid_voxels':valid_total,**c.metrics(counts)}
  c.save(out/'volume_metrics_partial.json',result)
  print('VOLUME',row['volume_id'],result[row['volume_id']]['dice'],flush=True)
 c.save(out/'result.json',{'status':'complete','model':a.model,'seed':a.seed,'smoke':a.smoke,'full':not a.smoke,'identity':infer.identity,
  'manifest_sha256':c.sha(R/'data/expanded_eval_manifest.json'),'volumes':result,'seconds':time.time()-start,'final_batch':infer.bs,
  'batch_reductions':infer.reductions,'reference_verification':verification,'threshold':.5,'tta':False,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'claim':'descriptive expansion; exposure categories separate'})
if __name__=='__main__':main()
