"""Exactly one 0.5 interpolation, GPU arithmetic; no parent writes."""
import gc,collections
import torch
from common import R,PA,MODEL,read,sha,atomic
def main():
 protocol=read(R/'protocol.json');parents=protocol['parents'];values=[]
 for name in ['D2-30k','D2-100k']:
  p=parents[name];assert sha(p['path'])==p['sha256'];v=torch.load(p['path'],map_location='cpu',mmap=True,weights_only=True)
  assert v['step']==(30000 if name=='D2-30k' else 100000);values.append(v)
 assert values[0]['official_weight_sha256']==values[1]['official_weight_sha256']
 a,b=[v['backbone'] for v in values];assert set(a)==set(b) and len(a)==368
 out={};nonfloat=[]
 with torch.inference_mode():
  for key in a:
   x,y=a[key],b[key];assert x.shape==y.shape and x.dtype==y.dtype
   if x.is_floating_point():
    u=x.to(device='cuda',dtype=torch.float64);v=y.to(device='cuda',dtype=torch.float64)
    assert torch.isfinite(u).all() and torch.isfinite(v).all()
    z=((u+v)*.5).to(x.dtype);assert torch.isfinite(z).all();out[key]=z.cpu();del u,v,z
   else:
    assert torch.equal(x,y),key;out[key]=x.clone();nonfloat.append(key)
 target=R/'checkpoints/Trajectory-Lerp05_ema_backbone.pth';target.parent.mkdir(exist_ok=True);assert not target.exists()
 torch.save({'backbone':out,'kind':'fixed_same_trajectory_EMA_interpolation','model':MODEL,'alpha':.5,'training_step':None,'parents':parents,'official_weight_sha256':values[0]['official_weight_sha256']},target)
 digest=sha(target);atomic(R/'encoder.json',{'id':MODEL,'path':str(target),'sha256':digest,'alpha':.5,'parents':parents,'keys':368,'nonfloating_keys':nonfloat,'dtypes':dict(collections.Counter(str(t.dtype) for t in out.values())),'arithmetic':'GPU float64 midpoint, cast to each original dtype','training_step':None,'SSL_training':False})
 for p in parents.values():assert sha(p['path'])==p['sha256']
 del values,a,b,out;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
