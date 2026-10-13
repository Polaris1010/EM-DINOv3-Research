"""Meaningful regression checks and two-update smoke; no diagnostic weight is exported."""
import os,sys,json,time
from pathlib import Path
from settings import *
sys.path.insert(0,str(R4))
import common as c,train as old
sys.path.insert(0,str(BASE))
from masked_data import supervised_target,MaskedPatches,loader
import torch,numpy as np
def main():
 out=Path(os.environ['RUN_OUTPUT']);start=time.time();torch.set_num_threads(2);c.seed(0)
 y=torch.tensor([[[1,-1],[0,-1]]]);a=torch.randn(1,2,2,2,requires_grad=True);v=old.loss(a,y);v.backward();g=a.grad.clone();assert torch.equal(g[:,:,0,1],torch.zeros_like(g[:,:,0,1])) and torch.equal(g[:,:,1,1],torch.zeros_like(g[:,:,1,1]))
 b=a.detach().clone();b[:,:,0,1]=torch.tensor([1000.,-1000.]);b[:,:,1,1]=torch.tensor([-1000.,1000.]);b.requires_grad_(True);w=old.loss(b,y);w.backward();assert torch.equal(v,w) and torch.equal(a.grad,b.grad)
 target=supervised_target(np.array([[4,0],[0,5]]),np.array([[13,13],[0,0]]));assert np.array_equal(target,[[1,0],[-1,-1]])
 # Loader integration: two workers, prefetch one, original source/augmentation sequence.
 batches=list(loader(0,16));assert len(batches)==2
 assert torch.isfinite(batches[0][0]).all();assert set(batches[0][1].unique().tolist())<={-1,0,1}
 ds=MaskedPatches(0);selected=[]
 for i,q in enumerate(ds.schedule):
  if q['volume_id'].startswith('han24_'):
   xx,yy,_=ds[i];assert (yy>=0).any();selected.append((xx,yy,q['volume_id']))
   if len(selected)==8:break
 assert len(selected)==8 and any((y==-1).any() for x,y,v in selected)
 net=c.model('P1',0);params=[p for p in net.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.98),weight_decay=.05);losses=[]
 for step in range(2):
  sample=selected[step*4:(step+1)*4];x=torch.stack([q[0] for q in sample]).cuda();y=torch.stack([q[1] for q in sample]).cuda();net.train();net.dino_encoder.eval();opt.zero_grad(set_to_none=True);logits=c.forward(net,x);v=old.loss(logits,y);assert torch.isfinite(v);v.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);assert all(p.grad is None for p in net.dino_encoder.parameters());torch.nn.utils.clip_grad_norm_(params,12,error_if_nonfinite=True);opt.step();losses.append(float(v.detach()))
 c.save(out/'result.json',dict(status='complete',diagnostic_only=True,steps=2,actual_samples=8,losses=losses,ignored_logit_invariance=True,ignored_pixel_gradients_exact_zero=True,valid_pixel_gradients_unchanged=True,region_ids_not_binary_labels=True,loader_workers=2,prefetch_factor=1,real_han_patches=8,trainable_gradients_finite=True,frozen_encoder_gradients_absent=True,weights_exported=False,seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
 print('MASKED_PIPELINE_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
