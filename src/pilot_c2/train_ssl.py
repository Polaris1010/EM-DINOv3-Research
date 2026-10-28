"""Fixed30k->45k continuation; production Barrier, resumable full state, no GT."""
from __future__ import annotations
import os,sys,json,time,hashlib,gc,random,argparse,signal
from pathlib import Path
ROOT=Path(os.environ['PILOT_C2_ROOT']);ASSET=ROOT.parent.parent;LATE=ROOT.parent/'ssl_diversity_d12_100k_20261018'
sys.path[:0]=[str(Path(__file__).parent),str(ASSET/'DINOv3/dinov3'),str(LATE/'code')]
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from dinov3.configs import get_default_config
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.fsdp.ac_compile_parallelize import activation_checkpoint_transformer
from dinov3.models import build_model_from_cfg
from ssl_full_data import Images,Collate
from runtime_guard import training_gpu_guard,website
import importlib.util
spec=importlib.util.spec_from_file_location('historical_ssl_trainer',LATE/'code/train_ssl.py');historical=importlib.util.module_from_spec(spec);spec.loader.exec_module(historical)
from production_adapter import ProductionBarrierAdapter
STOP=False
def stopping(*_):
 global STOP
 STOP=True
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def rng_hash():
 h=hashlib.sha256();h.update(repr(random.getstate()).encode());h.update(repr(np.random.get_state()).encode());h.update(torch.get_rng_state().numpy().tobytes());h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes());return h.hexdigest()
def optimizer_steps(opt):return sorted({int(x['step']) for x in opt.state.values()})
def main(model,resume):
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
 assert world==3;training_gpu_guard(world);torch.cuda.set_device(local);torch.set_num_threads(4)
 dist.init_process_group('nccl',device_id=torch.device('cuda',local));website()
 for sig in [signal.SIGTERM,signal.SIGUSR1]:signal.signal(sig,stopping)
 protocol=json.loads((ROOT/'protocol.json').read_text());assert model in protocol['models']
 assert protocol['lambda']==1 and protocol['margin']==.05 and protocol['reference_quantile']==.5 and protocol['Chebyshev_strictly_greater_than']==2 and protocol['layer']==23
 assert protocol['initial_step']==30000 and protocol['final_step']==45000 and protocol['additional_updates_per_model']==15000
 assert not protocol['diagnostic_VJP'] and not protocol['diagnostic_extra_student_forward']
 out=ROOT/'runs'/model;out.mkdir(parents=True,exist_ok=True)
 cfg=OmegaConf.merge(get_default_config(),OmegaConf.load(LATE/'D12_base.yaml'))
 assert cfg.train.batch_size_per_gpu==16 and cfg.crops.global_crops_size==256 and cfg.crops.local_crops_size==112 and not cfg.gram.use_loss
 seed=1000;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed);torch.backends.cudnn.benchmark=False
 with torch.device('meta'):net=SSLMetaArch(cfg)
 net.to_empty(device='cuda');natural=torch.load(ASSET/'DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',map_location='cpu',mmap=True,weights_only=True)
 net.student.backbone.load_state_dict(natural,strict=True);del natural
 net.student.dino_head.init_weights();net.student.ibot_head.init_weights();net.dino_loss.init_weights();net.ibot_patch_loss.init_weights();net.model_ema.load_state_dict(net.student.state_dict(),strict=True)
 activation_checkpoint_transformer(cfg,net.student.backbone)
 params=list(net.student.parameters());opt=torch.optim.AdamW(net.get_params_groups(),lr=float(cfg.optim.lr),betas=(float(cfg.optim.adamw_beta1),float(cfg.optim.adamw_beta2)),foreach=True)
 parent=ROOT.parent/'ssl_diversity_d12_20261017/runs/D2/formal/checkpoints/step000030000/full.pth'
 identity={'stage':'Pilot-C2','model':model,'initial_step':30000,'final_step':45000,'parent_full_sha256':protocol['parent_full_sha256'],'protocol_sha256':sha(ROOT/'protocol.json'),'trainer_sha256':sha(Path(__file__)),'barrier_enabled':model=='C1_Barrier_45k','global_batch':48,'seed':0,'world_size':3,'legacy_config_sha256':protocol['legacy_SSL_config_sha256'],'source_commit':json.loads((ROOT/'SOURCE.json').read_text())['code_commit']}
 selected=Path(resume) if resume else parent
 if resume:
  assert selected.resolve().is_relative_to(out.resolve());receipt=json.loads(selected.with_name('metadata.json').read_text());assert sha(selected)==receipt['sha256']
 saved=torch.load(selected,map_location='cpu',mmap=True,weights_only=False);start=int(saved['step'])
 if resume:assert saved['identity']==identity and 30000<start<45000
 else:assert start==30000 and saved['identity']['experiment_id']=='D2' and saved['identity']['manifest_sha256']==protocol['D2_manifest_sha256']
 net.student.load_state_dict(saved['student'],strict=True);net.model_ema.load_state_dict(saved['teacher'],strict=True);net.dino_loss.load_state_dict(saved['dino_loss'],strict=True);net.ibot_patch_loss.load_state_dict(saved['ibot_loss'],strict=True);opt.load_state_dict(saved['optimizer'])
 assert optimizer_steps(opt)==[start]
 student_exact=all(torch.equal(t,saved['student'][k]) for k,t in historical.full_state(net.student).items());teacher_exact=all(torch.equal(t,saved['teacher'][k]) for k,t in historical.full_state(net.model_ema).items());assert student_exact and teacher_exact
 prior_rng=saved['rng_by_rank'][rank];chain=saved['sample_chain'];del saved;gc.collect()
 anchor=None;adapter=None
 if model=='C1_Barrier_45k':
  with torch.device('meta'):anchor,dim=build_model_from_cfg(cfg,only_teacher=True)
  anchor.to_empty(device='cuda');v=torch.load(parent.with_name('teacher_backbone.pth'),map_location='cpu',mmap=True,weights_only=True);anchor.load_state_dict(v['backbone'],strict=True);del v
  anchor.requires_grad_(False).eval();assert dim==1024 and anchor.patch_size==16 and anchor.n_storage_tokens==4
  adapter=ProductionBarrierAdapter(net,anchor)
 net.train();dataset=Images(LATE/'D2_manifest.json',cfg,0,start*16,(45000-start)*16,rank=rank,world_size=3,batch_per_rank=16)
 loader=DataLoader(dataset,batch_size=16,num_workers=2,prefetch_factor=1,persistent_workers=False,pin_memory=False,multiprocessing_context='spawn',collate_fn=Collate(cfg),generator=torch.Generator().manual_seed(seed+rank))
 iterator=iter(loader);first_batch=next(iterator);historical.restore_rng(prior_rng)
 atomic(out/f'initial_rank{rank}_step{start}.json',{'identity':identity,'step':start,'student_strict_exact':student_exact,'teacher_strict_exact':teacher_exact,'optimizer_steps':optimizer_steps(opt),'RNG_restored_hash':rng_hash(),'sample_chain':chain,'production_mode':True,'extra_diagnostic_forward':False,'extra_VJP':False})
 t0=time.monotonic();completed=start;losses=[]
 def checkpoint(step):
  states=[None]*3;dist.all_gather_object(states,historical.rng_state())
  if rank==0:
   directory=out/'checkpoints'/f'step{step:09d}';directory.mkdir(parents=True,exist_ok=False);tmp=directory/'full.tmp';target=directory/'full.pth'
   torch.save({'format_version':1,'identity':identity,'step':step,'student':historical.full_state(net.student),'teacher':historical.full_state(net.model_ema),'dino_loss':historical.full_state(net.dino_loss),'ibot_loss':historical.full_state(net.ibot_patch_loss),'optimizer':opt.state_dict(),'rng_by_rank':states,'sample_chain':chain},tmp);tmp.replace(target);digest=sha(target)
   atomic(directory/'metadata.json',{'identity':identity,'step':step,'sha256':digest,'sample_chain':chain});atomic(out/'latest_checkpoint.json',{'path':str(target),'step':step,'sha256':digest})
   if step==45000:
    for label,module in [('teacher_backbone',net.model_ema.backbone),('student_backbone',net.student.backbone)]:
     p=directory/(label+'.pth');torch.save({'step':45000,'backbone':historical.full_state(module),'manifest_sha256':protocol['D2_manifest_sha256'],'official_weight_sha256':'8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035','config_sha256':protocol['legacy_SSL_config_sha256'],'C2_protocol_sha256':identity['protocol_sha256'],'model':model},p);atomic(p.with_suffix('.json'),{'path':str(p),'sha256':sha(p),'step':45000,'kind':label})
    p=directory/'teacher_backbone.pth';system=ROOT/'systems'/model;atomic(system/'encoder.json',{'path':str(p),'sha256':sha(p),'keys':368,'step':45000,'model':model,'EMA':True,'formal_additional_updates':15000})
  dist.barrier()
 with (out/f'steps_rank{rank}.jsonl').open('a') as log:
  for step in range(start,45000):
   flag=torch.tensor(int(STOP or (ROOT/'STOP').exists()),device='cuda');dist.all_reduce(flag,op=dist.ReduceOp.MAX)
   if int(flag):break
   if step%100==0:website()
   torch.cuda.synchronize();beg=time.monotonic();batch=first_batch if step==start else next(iterator);data_seconds=time.monotonic()-beg
   locations=batch.pop('_sample_locations');first=batch.pop('_first_index');digest=batch.pop('_sample_digest');assert first==step*48+rank*16
   hashes={k:tensor_hash(v) for k,v in batch.items() if torch.is_tensor(v)};digests=[None]*3;dist.all_gather_object(digests,digest)
   for item in digests:chain=hashlib.sha256(bytes.fromhex(chain)+bytes.fromhex(item)).hexdigest()
   batch['global_batch_size']=48;lr=historical.learning_rate(step,float(cfg.optim.lr),float(cfg.optim.min_lr))
   for group in opt.param_groups:group['lr']=lr*group['lr_multiplier'];group['weight_decay']=float(cfg.optim.weight_decay)*group['wd_multiplier']
   before_rng=rng_hash();opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=net.forward_backward(batch,teacher_temp=.07,iteration=step)
   assert torch.isfinite(loss);extra=adapter.finish_step() if adapter else {}
   historical.reduce_gradients(params,3);gn=torch.nn.utils.clip_grad_norm_(params,float(cfg.optim.clip_grad),error_if_nonfinite=True);opt.step();net.update_ema(.999);torch.cuda.synchronize();completed=step+1
   row={'model':model,'rank':rank,'step':completed,'resume_start':start,'loss':float(loss.detach()),'grad_norm':float(gn),'lr':lr,'weight_decay':.04,'global_batch':48,'first_index':first,'sample_digest':digest,'sample_chain':chain,'tensor_hashes':hashes,'before_model_RNG':before_rng,'after_model_RNG':rng_hash(),'data_seconds':data_seconds,'iteration_seconds':time.monotonic()-beg,'elapsed_seconds':time.monotonic()-t0,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'loss_components':{k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in metrics.items() if 'loss' in k},**extra}
   log.write(json.dumps(row)+'\n');log.flush();losses.append(row['loss'])
   if completed%20==0:atomic(out/f'progress_rank{rank}.json',{**{k:row[k] for k in ['model','rank','step','loss','lr','elapsed_seconds','iteration_seconds','peak_allocated_bytes']},'additional_completed':completed-30000,'additional_budget':15000,'barrier_raw':extra.get('barrier_raw'),'active_fraction':extra.get('active_fraction'),'barrier_backward_seen':extra.get('barrier_backward_seen'),'loss_median200':float(np.median(losses[-200:]))})
   if completed%1000==0:checkpoint(completed)
 if completed!=45000:
  if completed>start and completed%1000:checkpoint(completed)
  atomic(out/f'interrupted_rank{rank}.json',{'step':completed,'stopped_without_extension':True});dist.barrier();dist.destroy_process_group();return 75
 assert optimizer_steps(opt)==[45000]
 if anchor is not None:
  v=torch.load(parent.with_name('teacher_backbone.pth'),map_location='cpu',mmap=True,weights_only=True);assert all(torch.equal(t.cpu(),v['backbone'][k]) for k,t in anchor.state_dict().items());assert not any(p.grad is not None for p in anchor.parameters());del v
 tracked=next(net.student.backbone.parameters()).detach();copies=[torch.empty_like(tracked) for _ in range(3)];dist.all_gather(copies,tracked);delta=max(float((copies[0]-x).abs().max()) for x in copies);assert delta<=1e-6
 atomic(out/f'complete_rank{rank}.json',{'status':'complete','model':model,'rank':rank,'initial_step':30000,'completed_step':45000,'additional_updates':15000,'optimizer_steps':optimizer_steps(opt),'sample_chain':chain,'anchor_unchanged':True,'tracked_parameter_rank_max_difference':delta,'seconds':time.monotonic()-t0,'production_mode':True,'extra_diagnostic_forward':False,'extra_VJP':False})
 if rank==0:atomic(out/'result.json',{'status':'complete','model':model,'initial_step':30000,'completed_step':45000,'additional_updates':15000,'sample_chain':chain,'seconds':time.monotonic()-t0,'identity':identity})
 dist.barrier();dist.destroy_process_group();return 0
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['C0_Vanilla_45k','C1_Barrier_45k'],required=True);p.add_argument('--resume',type=Path);a=p.parse_args();raise SystemExit(main(a.model,a.resume))
