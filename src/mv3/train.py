"""Matched 40k frozen original Primus training; no historical checkpoint writes."""
import argparse,collections,gc,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from data import R,OLD,atomic,manifest,open_arrays,read,sha,Samples
sys.path.insert(0,str(OLD/'src'))
from mv_probe import build,forward,loss,sliding,confusion,overlap,head_state,state_hash

def gpu_guard():
 import csv,subprocess
 gpu={r[1].strip():int(r[0]) for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
 visible=os.environ.get('CUDA_VISIBLE_DEVICES');assert visible in gpu and gpu[visible] in [1,2,3],'Use exactly one permitted physical GPU UUID'
 assert torch.cuda.device_count()==1
 return gpu[visible]

@torch.no_grad()
def monitor(net,out):
 net.eval();rows=[]
 for row in manifest()['entries']:
  if row['supervised_role']!='dev':continue
  im,gt,valid,j=open_arrays(row);legal=j['valid_z'];zs=sorted({legal[i] for i in np.linspace(0,len(legal)-1,min(8,len(legal)),dtype=int)})
  c=np.zeros(4,np.int64)
  for z in zs:
   pred=sliding(net,im[z],tile_batch_size=16);v=valid[z].astype(bool) if valid is not None else np.ones(pred.shape,bool)
   c+=confusion(pred[v],gt[z].astype(bool)[v])
  rows.append({'region_id':row['id'],'source_id':row['source_id'],'slices':zs,**overlap(c)})
 means={s:float(np.mean([r['dice'] for r in rows if r['source_id']==s])) for s in sorted({r['source_id'] for r in rows})}
 assert len(means)==12
 return {'rows':rows,'source_means':means,'source_macro_dice':float(np.mean(list(means.values()))),'selection_used':False,'tta':False}

def save_checkpoint(path,net,opt,step,meta,counts,sampling_hash):
 temporary=path.with_suffix('.tmp')
 torch.save({'head':head_state(net),'optimizer':opt.state_dict(),'step':step,'metadata':meta,'source_draws':dict(counts),'sampling_hash':sampling_hash},temporary)
 temporary.replace(path)
 digest=sha(path);atomic(path.with_suffix('.json'),{'path':str(path),'sha256':digest,'step':step,'metadata':meta,'source_draws':dict(counts),'sampling_hash':sampling_hash});return digest

def smoke(model,seed,out):
 physical=gpu_guard();net,spec=build({'G-P0':'P0','G-B100':'B100k'}[model],seed)
 initial=state_hash(head_state(net));before=state_hash(net.dino_encoder.state_dict());sampler=Samples(seed);pars=[p for p in net.parameters() if p.requires_grad]
 opt=torch.optim.AdamW(pars,lr=1e-4,weight_decay=.05,betas=(.9,.98));rows=[];t0=time.time();sampling=hashlib.sha256()
 for i,source in enumerate(sampler.sources):
  net.train();net.dino_encoder.eval();x,y,coords=sampler.sample(i,forced_source=source);sampling.update(json.dumps(coords,sort_keys=True).encode())
  logits=forward(net,x);v=loss(logits,y);assert torch.isfinite(v)
  if source=='han24' and manifest().get('version')==2:
   assert all(h[2] is None and h[3]['valid_path'] is None for r,h in [(r,sampler.handles[r['id']]) for r in sampler.rows if r['source_id']==source])
   assert int((y>=0).sum())==int(y.numel()),'Han whole native crop must contribute to loss'
  # Padding and other sources retain the shared ignore-loss behavior.
  fixture=torch.randn(1,2,8,8,device='cuda',requires_grad=True);target=torch.randint(0,2,(1,8,8),device='cuda');target[:,:4,:]=-1
  a=loss(fixture,target);changed=fixture.detach().clone();changed[:,:,:4,:]+=torch.randn_like(changed[:,:,:4,:])*100
  assert torch.allclose(a.detach(),loss(changed,target),atol=1e-7,rtol=0)
  a.backward();assert torch.count_nonzero(fixture.grad[:,:,:4,:])==0
  opt.zero_grad(set_to_none=True);v.backward();assert all(p.grad is None for p in net.dino_encoder.parameters())
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pars)
  torch.nn.utils.clip_grad_norm_(pars,12,error_if_nonfinite=True);opt.step()
  rows.append({'source_id':source,'loss':float(v.detach()),'valid_pixels':int((y>=0).sum()),'coords':coords})
 assert before==state_hash(net.dino_encoder.state_dict())
 meta={'stage':'MV3_SMOKE_ONLY','model':model,'seed':seed,'decoder_initial_sha256':initial,'manifest_sha256':sha(R/'MV-Generalist.json'),'protocol_sha256':sha(R/'MV3_protocol.json'),'encoder_sha256':spec['sha256'],'encoder_frozen':True,'tta':False}
 path=out/'smoke_head.pth';save_checkpoint(path,net,opt,len(rows),meta,collections.Counter(r['source_id'] for r in rows),sampling.hexdigest())
 saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True);assert state_hash(saved['head'])==state_hash(head_state(net));assert saved['metadata']==meta
 net.eval();dev=[]
 for source in sampler.sources:
  row=next(r for r in manifest()['entries'] if r['source_id']==source and r['supervised_role']=='dev');im,gt,valid,j=open_arrays(row);z=j['valid_z'][len(j['valid_z'])//2]
  # Bounded real-data pipeline check, not a development performance result.
  h=min(256,im.shape[1]);w=min(256,im.shape[2]);pred=sliding(net,im[z,:h,:w],tile_batch_size=16)
  assert pred.shape==gt[z,:h,:w].shape
  if source=='han24' and manifest().get('version')==2:assert valid is None and j['valid_path'] is None
  dev.append({'source_id':source,'region_id':row['id'],'shape':list(pred.shape),'finite':True,'annotation_mask_used':valid is not None})
 j={'status':'PASS',**meta,'trainable_parameters':sum(p.numel() for p in pars),'encoder_unchanged':True,'ignore_loss_invariant':True,'ignore_gradient_zero':True,'checkpoint_roundtrip':True,'physical_gpu':physical,'train':rows,'dev_smoke':dev,'sampling_sha256':sampling.hexdigest(),'seconds':time.time()-t0,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
 atomic(out/'smoke.json',j);print('SMOKE_PASS',model,j['seconds'],flush=True)

def train(model,seed,out):
 physical=gpu_guard();protocol=read(R/'MV3_protocol.json');steps=protocol['optimizer_steps'];assert steps==40000
 net,spec=build(protocol['models'][model],seed);initial=state_hash(head_state(net));encoder_before=state_hash(net.dino_encoder.state_dict())
 sampler=Samples(seed);pars=[p for p in net.parameters() if p.requires_grad];opt=torch.optim.AdamW(pars,lr=protocol['lr'],weight_decay=protocol['weight_decay'],betas=tuple(protocol['betas']))
 meta={'stage':'MV-3','model':model,'encoder_model':protocol['models'][model],'seed':seed,'optimizer_steps':steps,'decoder_initial_sha256':initial,'encoder_sha256':spec['sha256'],'encoder_frozen':True,'gates':False,'tta':False,'interaction_indices':[5,11,17,23],'manifest_sha256':sha(R/'MV-Generalist.json'),'protocol_sha256':sha(R/'MV3_protocol.json'),'trainer_sha256':sha(__file__),'loader_sha256':sha(R/'src/data.py'),'primus_source_sha256':spec['primus_source_sha256'],'physical_gpu':physical,'training_batch_size':1,'selection':'fixed_step40000'}
 if (out/'run_protocol.json').exists():assert read(out/'run_protocol.json')==meta,'Changed protocol on an existing run'
 else:atomic(out/'run_protocol.json',meta)
 start=0;counts=collections.Counter();sampling_hash='0'*64;curve=[];recent=[];t0=time.time()
 checkpoints=sorted(out.glob('head_step*.pth'))
 if checkpoints:
  saved=torch.load(checkpoints[-1],map_location='cpu',weights_only=True,mmap=True);assert saved['metadata']==meta
  missing,extra=net.load_state_dict(saved['head'],strict=False);assert not extra and all(k.startswith('dino_encoder.') for k in missing)
  opt.load_state_dict(saved['optimizer']);start=saved['step'];counts.update(saved['source_draws']);sampling_hash=saved['sampling_hash'];del saved
  if (out/'curve.json').exists():curve=read(out/'curve.json')
 for i in range(start,steps):
  net.train();net.dino_encoder.eval();x,y,coords=sampler.sample(i);counts[coords['source']]+=1
  sampling_hash=hashlib.sha256(bytes.fromhex(sampling_hash)+json.dumps(coords,sort_keys=True).encode()).hexdigest()
  lr=protocol['lr']*min(1.,(i+1)/protocol['warmup_steps'])*(.1+.9*.5*(1+np.cos(np.pi*i/steps)))
  for g in opt.param_groups:g['lr']=lr
  opt.zero_grad(set_to_none=True);v=loss(forward(net,x),y);assert torch.isfinite(v);v.backward()
  assert all(p.grad is None for p in net.dino_encoder.parameters());torch.nn.utils.clip_grad_norm_(pars,protocol['gradient_clip_norm'],error_if_nonfinite=True);opt.step()
  recent.append(float(v.detach()));recent=recent[-200:]
  if (i+1)%100==0:
   atomic(out/'progress.json',{'model':model,'seed':seed,'stage':'train','step':i+1,'steps':steps,'loss_median200':float(np.median(recent)),'seconds':time.time()-t0,'source_draws':dict(counts),'sampling_hash':sampling_hash,'lr':lr})
  if (i+1)%1000==0:print('STEP',model,seed,i+1,float(np.median(recent)),time.time()-t0,flush=True)
  if i+1 in protocol['milestones']:
   digest=save_checkpoint(out/f'head_step{i+1:06d}.pth',net,opt,i+1,meta,counts,sampling_hash)
   atomic(out/'progress.json',{'model':model,'seed':seed,'stage':'development_monitor','step':i+1,'steps':steps,'checkpoint_sha256':digest})
   val=monitor(net,out);curve=[r for r in curve if r['step']!=i+1];curve.append({'step':i+1,'loss_median200':float(np.median(recent)),'checkpoint_sha256':digest,**val});atomic(out/'curve.json',curve)
   print('MILESTONE',model,seed,i+1,val['source_macro_dice'],flush=True)
 assert encoder_before==state_hash(net.dino_encoder.state_dict())
 atomic(out/'training_complete.json',{'status':'complete',**meta,'completed_steps':steps,'source_draws':dict(counts),'sampling_hash':sampling_hash,'encoder_unchanged':True,'seconds':time.time()-t0,'curve':curve,'convergence_assessment':'report common40k curve; no automatic budget extension'})
 del net,opt;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','train']);ap.add_argument('--model',choices=['G-P0','G-B100'],required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
 out=R/('smoke' if a.mode=='smoke' else 'runs')/a.model/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
 (smoke if a.mode=='smoke' else train)(a.model,a.seed,out)
