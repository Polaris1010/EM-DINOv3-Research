import os,json,sys
from pathlib import Path
import torch,numpy as np
def main():
 a=torch.load(sys.argv[1],map_location='cpu',mmap=True,weights_only=False);b=torch.load(sys.argv[2],map_location='cpu',mmap=True,weights_only=False);maximum=0.;count=0
 def compare(x,y,path):
  nonlocal maximum,count
  if isinstance(x,np.ndarray):assert np.array_equal(x,y),path
  elif torch.is_tensor(x):
   assert x.shape==y.shape and x.dtype==y.dtype,path
   if x.is_floating_point():
    d=float((x-y).abs().max()) if x.numel() else 0.;maximum=max(maximum,d);assert torch.equal(x,y),f'{path}: {d}'
   else:assert torch.equal(x,y),path
   count+=1
  elif isinstance(x,dict):
   assert x.keys()==y.keys(),path
   for k in x:compare(x[k],y[k],path+'/'+str(k))
  elif isinstance(x,(list,tuple)):
   assert len(x)==len(y),path
   for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'/'+str(i))
  else:assert x==y,path
 for k in ['teacher','student','gram_teacher','dino_loss','ibot_loss','optimizer','completed_steps','anchor_updates','consumption_chain','next_sample','config','rng','fixed_reference']:compare(a[k],b[k],k)
 out=Path(os.environ['RUN_OUTPUT']);(out/'result.json').write_text(json.dumps(dict(status='complete',compared_tensors=count,max_abs_difference=maximum,atol=1e-6,rtol=1e-6)))
if __name__=='__main__':main()
