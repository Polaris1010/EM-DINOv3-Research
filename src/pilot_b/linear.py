"""Matched Pilot A Linear23; descriptive result never gates Primus training."""
import gc,hashlib,json,sys,time,csv,os
import numpy as np
import torch
import torch.nn.functional as F
from common import R,PA,A2,MODEL,read,sha,atomic,module
def csvout(name,rows):
 p=R/'linear'/name
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 torch.set_num_threads(2);torch.manual_seed(20261027);started=time.time();out=R/'linear';out.mkdir(exist_ok=True);assert not (out/'probe.pth').exists()
 p=read(PA/'protocol.json');items=read(PA/'samples.json');m=read(R/'encoder.json');assert sha(m['path'])==m['sha256']
 arch=p['models'][0];sys.path.insert(0,arch['dinov3_root'])
 from omegaconf import OmegaConf
 from dinov3.models import build_model_from_cfg
 enc,dim=build_model_from_cfg(OmegaConf.load(arch['resolved_config_path']),only_teacher=True);v=torch.load(m['path'],weights_only=True,mmap=True,map_location='cpu')
 enc.load_state_dict(v['backbone'],strict=True,assign=True);enc.requires_grad_(False).eval().cuda();assert dim==1024 and enc.n_storage_tokens==4
 images=np.load(PA/'images.npy',mmap_mode='r');n=len(items);cache=np.lib.format.open_memmap(out/'layer23.npy',mode='w+',dtype=np.float16,shape=(n,256,1024))
 with torch.inference_mode():
  for first in range(0,n,8):
   x=torch.tensor(np.array(images[first:first+8],copy=True),device='cuda',dtype=torch.float32);x=((x/255-.5)/.25)[:,None].repeat(1,3,1,1)
   with torch.autocast('cuda',dtype=torch.bfloat16):features=enc.get_intermediate_layers(x,n=[23],reshape=False,return_class_token=True,return_extra_tokens=True,norm=True)
   z,cls,register=features[0];assert z.shape[1:]==(256,1024) and register.shape[1:]==(4,1024) and torch.isfinite(z).all()
   cache[first:first+len(x)]=z.float().cpu().numpy()
 cache.flush();del enc,v;gc.collect();torch.cuda.empty_cache()
 labels=np.load(PA/'train_labels.npz');train=np.array([i['role']=='train' for i in items]);sources=sorted({i['source'] for i in items if i['role']=='train'})
 valid=labels['valid'][train].reshape(-1);target=labels['fraction'][train].reshape(-1);source_ids=np.repeat([sources.index(i['source']) for i in items if i['role']=='train'],256)
 usable=valid&((target==0)|(target>=.5));binary=target>=.5;bins={(s,c):np.flatnonzero(usable&(source_ids==s)&(binary==c)) for s in range(len(sources)) for c in [False,True]}
 available=[s for s in range(len(sources)) if len(bins[s,False]) and len(bins[s,True])];assert len(available)==12
 rng=np.random.default_rng(20261027);batches=[]
 for _ in range(500):
  si=rng.choice(available,4096);cl=rng.integers(2,size=len(si));idx=np.empty(len(si),np.int64)
  for s in available:
   for c in [0,1]:
    select=(si==s)&(cl==c);pool=bins[s,bool(c)];idx[select]=pool[rng.integers(len(pool),size=select.sum())]
  batches.append(idx)
 batchhash=hashlib.sha256(np.stack(batches).tobytes()).hexdigest();assert batchhash==read(PA/'models/D2-30k/probe_complete.json')['coordinate_batch_hash']
 x=torch.tensor(np.array(cache[train],copy=True).reshape(-1,1024),device='cuda',dtype=torch.float32);sid=torch.tensor(source_ids,device='cuda');weights=torch.zeros(len(x),device='cuda')
 vv=torch.tensor(valid,device='cuda')
 for s in range(len(sources)):
  mask=(sid==s)&vv;weights[mask]=1/max(1,int(mask.sum()))/len(sources)
 mean=(x*weights[:,None]).sum(0);var=((x-mean).square()*weights[:,None]).sum(0);scale=var.sqrt().clamp_min(1e-4);x=(x-mean)/scale
 head=torch.nn.Linear(1024,1,device='cuda');torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias);opt=torch.optim.AdamW(head.parameters(),lr=.03,weight_decay=.001)
 y=torch.tensor(binary.astype(np.float32),device='cuda')
 for indices in batches:
  idx=torch.as_tensor(indices,device='cuda');opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(head(x[idx]).squeeze(-1),y[idx]);assert torch.isfinite(loss);loss.backward();opt.step()
 state={'weight':head.weight.detach().cpu(),'bias':head.bias.detach().cpu(),'mean':mean.cpu(),'scale':scale.cpu()};torch.save(state,out/'probe.pth')
 probability=np.lib.format.open_memmap(out/'linear_probability.npy',mode='w+',dtype=np.float32,shape=(n,256))
 with torch.inference_mode():
  for first in range(0,n,16):
   z=torch.tensor(np.array(cache[first:first+16],copy=True),device='cuda',dtype=torch.float32);probability[first:first+len(z)]=head((z-mean)/scale).sigmoid().squeeze(-1).cpu().numpy()
 probability.flush();probehash=sha(out/'probe.pth')
 atomic(out/'fit_complete.json',{'status':'complete','steps':500,'seed':20261027,'batch':4096,'lr':.03,'weight_decay':.001,'coordinate_batch_hash':batchhash,'probe_sha256':probehash,'encoder_sha256':m['sha256'],'samples_sha256':sha(PA/'samples.json'),'evaluation_GT_read':False,'CLS_and_registers_excluded':True,'cache_dtype':'float16','source_normalization_and_sampler':'identical Pilot A','Primus_gate':False})
 del x,opt,head;torch.cuda.empty_cache()
 # Evaluation starts only after the train-only probe SHA is fixed.
 labels=np.load(PA/'evaluation_labels.npz');os.environ.setdefault('PILOT_A2_ROOT',str(A2));helpers=module('pilot_b_a2_metrics',A2/'code/common.py');stats=module('pilot_b_spectrum',PA/'code/stats.py')
 metrics=[];spectra=[];distributions=[]
 for si,source in enumerate(p['diagnostic_sources']):
  ids=[i['id'] for i in items if i['source']==source and i['subset']=='random'];assert len(ids)==64
  prob=torch.tensor(np.array(probability[ids],copy=True),device='cuda').reshape(-1,1,16,16);native=F.interpolate(prob,size=(256,256),mode='bilinear',align_corners=False)[:,0]
  gt=torch.tensor(labels['GT'][ids],device='cuda');v=torch.tensor(labels['valid_pixels'][ids],device='cuda');far=torch.tensor(labels['far_pixels'][ids],device='cuda')
  c=helpers.counts(native,gt,v,far,0);r=helpers.Ranked(native[v],gt[v],torch.zeros(int(v.sum()),device='cuda',dtype=torch.int64));auc,ap,_=r.calculate()
  metrics.append({'model':MODEL,'source':source,'crops':64,**c,**helpers.summarize(c),'pixel_AUROC':auc,'pixel_AP':ap,'probe_sha256':probehash})
  raw=torch.tensor(np.array(cache[ids],copy=True).reshape(-1,1024),device='cuda',dtype=torch.float32);candidates=np.flatnonzero(labels['valid'][ids].reshape(-1))
  for repeat in range(3):
   generator=np.random.default_rng(np.random.SeedSequence([20261027,si,repeat]));chosen=generator.choice(candidates,512,replace=False);d=stats.spectrum(raw[torch.tensor(chosen,device='cuda')]);eigen=d.pop('normalized_eigenvalues')
   spectra.append({'model':MODEL,'source':source,'layer':23,'repeat':repeat,'patches':512,**d});distributions.append({'source':source,'repeat':repeat,'variance_fractions_descending':json.dumps(eigen.tolist())})
  atomic(out/'progress.json',{'source':source,'stage':'frozen_linear_evaluation','seconds':time.time()-started});del raw,r;torch.cuda.empty_cache()
 assert probehash==sha(out/'probe.pth') and m['sha256']==sha(m['path'])
 csvout('pixel_metrics.csv',metrics);csvout('spectrum.csv',spectra);csvout('spectrum_distribution.csv',distributions)
 atomic(out/'complete.json',{'status':'complete','seconds':time.time()-started,'train_crops':312,'random_eval_crops':320,'historical_error_case_excluded':True,'Primus_training_is_unconditional':True,'probe_sha256':probehash})
if __name__=='__main__':main()
