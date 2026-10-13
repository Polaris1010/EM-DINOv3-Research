"""Paired, deterministic local continuation of the official DINOv3 Gram recipe."""
import os,sys,json,csv,random,time,signal,argparse,gc,hashlib,math
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=Path('/opt/data/3_117/tanghr');SRC=ROOT/'DINOv3/dinov3'
sys.path.insert(0,str(ROOT/'codex/mitotopo_20260908_round2/deps'));sys.path.insert(0,str(SRC))
import numpy as np,torch
from omegaconf import OmegaConf
from dinov3.configs import get_default_config
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.data import DataAugmentationDINO
from dinov3.data.collate import collate_data_and_cast
from dinov3.data.masking import MaskingGenerator
from torch.utils.data import Dataset,DataLoader
from PIL import Image
EM=ROOT/'DINOv3/med_checkpoints_continual_v2/eval/training_124999/teacher_checkpoint.pth'
OFFSET=1000000;STOP=False
def stopped(*_):
 global STOP
 STOP=True
def savejson(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=str));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def config(weight,batch):
 cfg=OmegaConf.merge(get_default_config(),OmegaConf.load(ROOT/'codex/mitotopo_20260908_round2/official/gram_anchor_official.yaml'))
 old=OmegaConf.load(ROOT/'DINOv3/med_checkpoints_continual_v2/config.yaml')
 cfg.student=old.student;cfg.student.pretrained_weights='';cfg.student.resume_from_teacher_chkpt=str(EM)
 for head in ['dino','ibot']:
  for k,v in old[head].items():
   if k.startswith('head_'):cfg[head][k]=v
 cfg.gram.ckpt=str(EM);cfg.gram.use_loss=True;cfg.gram.loss_weight_schedule.end=weight
 cfg.train.batch_size_per_gpu=batch;cfg.train.num_workers=4;cfg.train.cache_dataset=False;cfg.train.compile=False;cfg.train.sharded_eval_checkpoint=False
 return cfg
class Images(Dataset):
 def __init__(self,cfg,start,count):
  self.start=start;self.count=count;self.rows=list(csv.DictReader(open(BASE/'ssl_dataset_manifest.csv')));self.schedule=np.load(BASE/'ssl_schedule.npy',mmap_mode='r')
  self.aug=DataAugmentationDINO(global_crops_scale=cfg.crops.global_crops_scale,local_crops_scale=cfg.crops.local_crops_scale,local_crops_number=cfg.crops.local_crops_number,global_crops_size=cfg.crops.global_crops_size,local_crops_size=cfg.crops.local_crops_size,gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,horizontal_flips=cfg.crops.horizontal_flips,mean=cfg.crops.rgb_mean,std=cfg.crops.rgb_std)
 def __len__(self):return self.count
 def __getitem__(self,i):
  i+=self.start;j,s=map(int,self.schedule[i]);random.seed(s);np.random.seed(s);torch.manual_seed(s)
  with Image.open(self.rows[j]['path']) as f:im=f.convert('RGB')
  return self.aug(im),(i,j,s,hashlib.sha256(im.tobytes()).hexdigest())
class Collate:
 def __init__(self,cfg):self.cfg=cfg
 def __call__(self,samples):
  metadata=[x[1] for x in samples];i=metadata[0][0]
  random.seed(900000000+i);np.random.seed(900000000+i);torch.manual_seed(900000000+i)
  b=collate_data_and_cast([(x[0],x[1][0]) for x in samples],mask_ratio_tuple=self.cfg.ibot.mask_ratio_min_max,mask_probability=self.cfg.ibot.mask_sample_probability,dtype=torch.float32,n_tokens=256,mask_generator=MaskingGenerator(input_size=(16,16),max_num_patches=128))
  h=hashlib.sha256(json.dumps(metadata,separators=(',',':')).encode())
  for k,v in sorted(b.items()):
   if torch.is_tensor(v):h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
  b['_digest']=h.hexdigest();b['_first']=i
  return b
def clean(sd):return {k.replace('_checkpoint_wrapped_module.',''):v.detach().cpu() for k,v in sd.items()}
def rngstate():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--weight',type=float,required=True);ap.add_argument('--batch',type=int,default=16);ap.add_argument('--steps',type=int,required=True);ap.add_argument('--resume',type=Path);ap.add_argument('--cpu',action='store_true');a=ap.parse_args()
 assert a.weight in [0,2] and a.batch in [8,16] and a.steps<=30000
 out=Path(os.environ['RUN_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
 from snapshot import snapshot
 snapshot(out)
 random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 cfg=config(a.weight,a.batch);OmegaConf.save(cfg,out/'resolved_config.yaml');locked=json.load(open(BASE/'locked_inputs.json'))
 if a.cpu:
  from dinov3.train.cosine_lr_scheduler import linear_warmup_cosine_decay
  from dinov3.loss.gram_loss import GramLoss
  s=linear_warmup_cosine_decay(0,0,a.weight,1000000,1200000,1000);ds=Images(cfg,0,2);c=Collate(cfg);b=c([ds[0],ds[1]]);d=c([ds[0],ds[1]]);assert b['_digest']==d['_digest']
  x=torch.randn(2,16,32,requires_grad=True);y=torch.randn(2,16,32);loss=GramLoss(apply_norm=True,remove_neg=False,remove_only_teacher_neg=False);assert loss(x,x.detach()).item()==0;v=loss(x,y);v.backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
  savejson(out/'result.json',dict(status='complete',schedule={str(i):float(s[OFFSET+i]) for i in [0,1,500,999,1000,10000]},batch_shapes={k:list(v.shape) for k,v in b.items() if torch.is_tensor(v)},deterministic_batch=True,gram_gradient=True));return
 signal.signal(signal.SIGUSR1,stopped);signal.signal(signal.SIGTERM,stopped)
 torch.distributed.init_process_group('nccl',init_method='file://'+str(out/f'dist_init_{os.getpid()}'),rank=0,world_size=1)
 resume=torch.load(a.resume,map_location='cpu',mmap=True,weights_only=False) if a.resume else None
 start=int(resume['completed_steps']) if resume else 0;assert start<a.steps
 if resume:assert resume['config']==OmegaConf.to_container(cfg) and resume['schedule_sha256']==locked['schedule_sha256']
 net=SSLMetaArch(cfg);net.to_empty(device='cuda');ck=resume['teacher'] if resume else torch.load(EM,map_location='cpu',mmap=True,weights_only=True)['teacher']
 net.student.load_state_dict(resume['student'] if resume else ck,strict=True);net.model_ema.load_state_dict(ck,strict=True)
 net.gram_teacher.load_state_dict(resume['gram_teacher'] if resume else {k:v for k,v in ck.items() if k.startswith('backbone.')},strict=True);net.gram_teacher_initialized=True
 net.dino_loss.init_weights();net.ibot_patch_loss.init_weights()
 if resume:
  net.dino_loss.load_state_dict(resume['dino_loss']);net.ibot_patch_loss.load_state_dict(resume['ibot_loss'])
 del ck
 from dinov3.fsdp.ac_compile_parallelize import activation_checkpoint_transformer
 activation_checkpoint_transformer(cfg,net.student.backbone)
 groups=net.get_params_groups();lr=float(cfg.schedules.lr.end)*4*math.sqrt(a.batch/1024)
 for g in groups:g['lr']=lr*g['lr_multiplier'];g['weight_decay']=.04*g['wd_multiplier']
 opt=torch.optim.AdamW(groups,lr=lr,betas=(.9,.99),foreach=True)
 if resume:opt.load_state_dict(resume['optimizer'])
 reference=resume.get('fixed_reference') if resume else None
 saved_rng=resume['rng'] if resume else None;anchor_updates=resume['anchor_updates'] if resume else 0;chain=resume['consumption_chain'] if resume else '00'*32
 del resume;gc.collect();net.train();completed=start;t0=time.time();status='running';last={};hook={}
 def gram_hook(module,args,output):
  if output.requires_grad:output.register_hook(lambda grad:hook.update(gram_backward_coefficient=float(grad.detach())))
 net.gram_loss.register_forward_hook(gram_hook)
 fixed=[]
 for row in list(csv.DictReader(open(BASE/'ssl_dataset_manifest.csv')))[:4]:
  with Image.open(row['path']) as im:arr=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC)).copy()
  fixed.append(torch.from_numpy(arr).permute(2,0,1).float()/255)
 fixed=torch.stack(fixed).cuda();fixed=(fixed-torch.tensor(list(cfg.crops.rgb_mean),device='cuda')[None,:,None,None])/torch.tensor(list(cfg.crops.rgb_std),device='cuda')[None,:,None,None]
 def features():
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):return net.model_ema.backbone(fixed,is_training=True)['x_norm_patchtokens'].float().cpu()
 if reference is None:
  if start:
   held=clean(net.model_ema.state_dict());initial=torch.load(EM,map_location='cpu',mmap=True,weights_only=True)['teacher'];net.model_ema.load_state_dict(initial,strict=True)
   reference=features();net.model_ema.load_state_dict(held,strict=True);del held,initial;gc.collect()
  else:reference=features()
 def diagnostics():
  f=features();u=torch.nn.functional.normalize(f,dim=-1);v=torch.nn.functional.normalize(reference,dim=-1)
  with torch.no_grad():ema_delta=float(sum((x-y).square().sum() for x,y in zip(net.student.parameters(),net.model_ema.parameters())).sqrt())
  row=dict(step=completed,ema_parameter_l2_difference=ema_delta,patch_variance=float(f.var(dim=1).mean()),mean_token_norm=float(f.norm(dim=-1).mean()),cosine_to_P1=float((u*v).sum(-1).mean()),gram_mse_to_P1=float(((u@u.transpose(-1,-2))-(v@v.transpose(-1,-2))).square().mean()))
  with (out/'fixed_feature_diagnostics.jsonl').open('a') as fp:fp.write(json.dumps(row)+'\n')
 def checkpoint():
  payload=dict(format_version=2,teacher=clean(net.model_ema.state_dict()),student=clean(net.student.state_dict()),gram_teacher=clean(net.gram_teacher.state_dict()),dino_loss=clean(net.dino_loss.state_dict()),ibot_loss=clean(net.ibot_patch_loss.state_dict()),optimizer=opt.state_dict(),rng=rngstate(),completed_steps=completed,anchor_updates=anchor_updates,consumption_chain=chain,next_sample=completed*a.batch,schedule_sha256=locked['schedule_sha256'],manifest_sha256=locked['manifest_sha256'],config=OmegaConf.to_container(cfg),schedule_iteration=OFFSET+completed,base_lr=lr,teacher_temp=.07,ema=.999,status=status)
  payload['fixed_reference']=reference
  p=out/f'checkpoint_step{completed}.pth';tmp=p.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(p)
  index=dict(path=str(p),step=completed,sha256=sha(p))
  savejson(out/'latest_checkpoint.json',index)
  with (out/'checkpoint_index.jsonl').open('a') as fp:fp.write(json.dumps(index)+'\n')
  if completed in [0,1000,5000,10000,20000,30000]:
   ep=out/f'teacher_step{completed}.pth';torch.save(dict(teacher=payload['teacher'],completed_steps=completed,weight=a.weight,P1_sha256=locked['P1_sha256']),ep);savejson(ep.with_suffix('.json'),dict(sha256=sha(ep),step=completed,weight=a.weight))
  permanent={0,10,11,500,1000,5000,10000,20000,30000};roll=sorted([(int(p.stem.removeprefix('checkpoint_step')),p) for p in out.glob('checkpoint_step*.pth') if int(p.stem.removeprefix('checkpoint_step')) not in permanent])
  for _,old in roll[:-3]:old.unlink()
 if start==0:diagnostics();checkpoint()
 ds=Images(cfg,start*a.batch,(a.steps-start)*a.batch);dl=DataLoader(ds,batch_size=a.batch,num_workers=4,prefetch_factor=1,persistent_workers=False,pin_memory=False,multiprocessing_context='spawn',collate_fn=Collate(cfg),generator=torch.Generator().manual_seed(0))
 it=iter(dl)
 if saved_rng:restore(saved_rng)
 with (out/'steps.jsonl').open('w') as log:
  for step in range(start,a.steps):
   if STOP:status='interrupted';break
   b=next(it);assert b.pop('_first')==step*a.batch;digest=b.pop('_digest');chain=hashlib.sha256(bytes.fromhex(chain)+bytes.fromhex(digest)).hexdigest();b['global_batch_size']=a.batch
   opt.zero_grad(set_to_none=True);hook.clear()
   with torch.autocast('cuda',dtype=torch.bfloat16):total,values=net.forward_backward(b,teacher_temp=.07,iteration=OFFSET+step)
   assert torch.isfinite(total);grad=torch.nn.utils.clip_grad_norm_(net.student.parameters(),30,error_if_nonfinite=True);opt.step();net.update_ema(.999);completed=step+1
   if completed in [10000,20000,30000]:
    net.update_gram();anchor_updates+=1
    assert all(torch.equal(x,y) for x,y in zip(net.gram_teacher.backbone.parameters(),net.model_ema.backbone.parameters())),'Anchor refresh did not copy EMA'
    with (out/'anchor_updates.jsonl').open('a') as f:f.write(json.dumps(dict(completed_steps=completed,anchor_updates=anchor_updates,source='current EMA',affects_future_steps=completed<30000))+'\n')
   last=dict(step=completed,total=float(total.detach()),grad_norm=float(grad),base_lr=lr,teacher_temp=.07,ema=.999,consumption_chain=chain,batch_digest=digest,anchor_updates=anchor_updates,seconds=time.time()-t0,peak_gpu_bytes=torch.cuda.max_memory_allocated(),**hook,**{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in values.items()})
   assert last.get('gram_backward_coefficient',0)==0 if a.weight==0 else True
   log.write(json.dumps(last)+'\n');log.flush()
   if completed%10==0:print(json.dumps(last),flush=True);savejson(out/'progress.json',dict(status='running',target_steps=a.steps,**last))
   if completed%1000==0:diagnostics()
   if completed%1000==0 or completed in [10,11,500] or STOP:checkpoint()
  else:status='complete'
 del it,dl
 checkpoint();savejson(out/'result.json',dict(status=status,completed_steps=completed,target_steps=a.steps,weight=a.weight,batch=a.batch,**last));savejson(out/'progress.json',dict(status=status,target_steps=a.steps,**last));torch.distributed.destroy_process_group()
 if status!='complete':raise SystemExit(75)
if __name__=='__main__':main()
