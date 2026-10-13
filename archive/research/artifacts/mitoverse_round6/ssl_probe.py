"""Round4-matched probes; X1 enables autograd only in the last four encoder blocks."""
import os,sys,json,time,signal,argparse,math,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent;R4=BASE.parent/'mitoverse_round4_20260909_103239'
sys.path.insert(0,str(R4))
import common as c
import train as old
sys.path.insert(0,str(BASE))
from masked_data import loader
import torch,numpy as np
STOP=False
def stop(*_):
 global STOP
 STOP=True
def forward(net,x):
 with torch.autocast('cuda',dtype=torch.bfloat16):fs=net.dino_encoder.get_intermediate_layers(x,n=net.interaction_indices,reshape=True)
 return net.up_projection(torch.cat([g(f.float()) for g,f in zip(net.last_gates,fs)],dim=1))
def encoder_part(net):return {k:v.detach().cpu().clone() for k,v in net.dino_encoder.state_dict().items() if any(k.startswith(f'blocks.{i}.') for i in range(20,24))}
def load(net,ck,x1):
 c.load_head(net,ck['head'])
 if x1:
  missing,unexpected=net.dino_encoder.load_state_dict(ck['encoder_last4'],strict=False);assert not unexpected;assert all(not any(k.startswith(f'blocks.{i}.') for i in range(20,24)) for k in missing)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--encoder',type=Path,default=c.PATHS['P1']);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--x1',action='store_true');ap.add_argument('--steps',type=int,default=10000);ap.add_argument('--evaluate-from',type=Path);ap.add_argument('--validation-only',action='store_true');ap.add_argument('--legacy-only',action='store_true');a=ap.parse_args()
 out=Path(os.environ['RUN_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
 import shutil
 shutil.copytree(BASE/'exploratory_source_snapshot',out/'source_snapshot')
 shutil.copyfile(__file__,out/'source_snapshot'/'ssl_probe.py')
 c.seed(a.seed);c.PATHS['P1']=a.encoder
 assert not a.x1 and a.steps in [10,10000]
 if a.evaluate_from:assert a.validation_only != a.legacy_only
 signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
 net=c.model('P1',a.seed);fw=forward if a.x1 else c.forward;old.forward=fw
 hp=[p for p in net.parameters() if p.requires_grad];ep=[]
 if a.x1:
  for block in net.dino_encoder.blocks[20:24]:block.requires_grad_(True);ep.extend(block.parameters())
 assert len(net.dino_encoder.blocks)==24
 frozen=[p for p in net.dino_encoder.parameters() if not p.requires_grad];params=hp+ep
 identity=dict(encoder_path=str(a.encoder),encoder_sha256=c.sha(a.encoder),seed=a.seed,x1=a.x1,norm='N2',head_init_sha256=c.sha(R4/f'shared_head_init_seed{a.seed}.pth'),schedule_sha256=c.sha(BASE/f'train_patch_schedule_masked_seed{a.seed}.jsonl'),head_lr=1e-4,last4_lr=3e-6 if a.x1 else 0)
 c.save(out/'identity.json',identity)
 if a.evaluate_from:
  ck=torch.load(a.evaluate_from/'head_best.pth',map_location='cpu',weights_only=False);assert ck['identity']==identity;load(net,ck,a.x1)
  result=old.evaluate(net,'N2',out,['validation'] if a.validation_only else ['test1','test2','test3'],True)
  c.save(out/'metrics.json',dict(status='complete',exploratory=True,selected_step=ck['step'],**identity,**result))
  c.save(out/'result.json',dict(status='complete',exploratory=True,full_validation=a.validation_only,**result))
  return
  prefix='_'.join(out.name.split('_')[:2]);required=[BASE/'jobs'/f'{prefix}_{arm}_30000_head{s}_eval'/'metrics.json' for arm in ['G','C'] for s in range(3)]+[BASE/'jobs'/f'{prefix}_X1_seed{s}_eval'/'metrics.json' for s in range(3)]
  if all(p.exists() for p in required):
   import runpy
   sys.argv=[str(BASE/'report.py'),prefix];runpy.run_path(str(BASE/'report.py'),run_name='__main__')
  return
 opt=torch.optim.AdamW([dict(params=hp,lr=1e-4,base_lr=1e-4)]+([dict(params=ep,lr=3e-6,base_lr=3e-6)] if ep else []),betas=(.9,.98),weight_decay=.05)
 initial={k:p.detach().cpu().clone() for k,p in net.dino_encoder.named_parameters() if p.requires_grad};best=-1;curve=[];completed=0;t0=time.time();digest=hashlib.sha256();last={}
 def checkpoint(name):
  ck=dict(head=c.head(net),encoder_last4=encoder_part(net) if a.x1 else {},optimizer=opt.state_dict(),step=completed,identity=identity,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=__import__('random').getstate(),consumed_indices_sha256=digest.hexdigest())
  p=out/name;tmp=p.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(p);return ck
 data=loader(a.seed,a.steps*8,'N2')
 for step,(x,y,indices) in enumerate(data,1):
  if STOP:break
  assert indices.tolist()==list(range((step-1)*8,step*8));digest.update(indices.numpy().tobytes());net.train();net.dino_encoder.eval();x=x.cuda();y=y.cuda()
  scale=min(1.,step/50)*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/10000)))
  for g in opt.param_groups:g['lr']=g['base_lr']*scale
  opt.zero_grad(set_to_none=True);q=fw(net,x);v=old.loss(q,y);assert torch.isfinite(v);v.backward();assert all(p.grad is None for p in frozen);assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
  gn=torch.nn.utils.clip_grad_norm_(params,12,error_if_nonfinite=True);opt.step();completed=step
  if step==1 or step%25==0:
   last=dict(step=step,loss=float(v.detach()),grad_norm=float(gn),head_lr=opt.param_groups[0]['lr'],last4_lr=opt.param_groups[-1]['lr'] if ep else 0,seconds=time.time()-t0,peak_gpu_bytes=torch.cuda.max_memory_allocated());c.save(out/'progress.json',last);print(json.dumps(last),flush=True)
  if step%1000==0 or step==a.steps or STOP:
   ck=checkpoint('checkpoint_latest.pth')
   if a.steps>=1000 and not STOP:
    val=old.evaluate(net,'N2',out,['validation']);curve.append(dict(step=step,**val));c.save(out/'curve.json',curve)
    if val['source_macro_dice']>best:best=val['source_macro_dice'];torch.save({k:v for k,v in ck.items() if k!='optimizer'},out/'head_best.pth')
   testx=x[:1].detach().clone()
 del data
 if STOP:
  checkpoint('interrupted.pth');raise SystemExit(75)
 changes={k:float((p.detach().cpu()-initial[k]).abs().max()) for k,p in net.dino_encoder.named_parameters() if k in initial}
 if a.x1:assert all(any(v>0 for k,v in changes.items() if k.startswith(f'blocks.{i}.')) for i in range(20,24))
 net.eval()
 with torch.inference_mode():before=fw(net,testx).cpu()
 ck=torch.load(out/'checkpoint_latest.pth',map_location='cpu',weights_only=False);load(net,ck,a.x1)
 with torch.inference_mode():after=fw(net,testx).cpu()
 assert torch.equal(before,after)
 c.save(out/'result.json',dict(status='complete',steps=completed,**(identity | last),exploratory=True,actual_samples=completed*8,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),gradient_finite=True,frozen_grad_absent=True,last4_max_changes=changes,reload_exact=True,best_validation_dice=best))
if __name__=='__main__':main()
