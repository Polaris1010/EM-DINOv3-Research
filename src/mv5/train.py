"""Bounded MitoNet fine-tuning using the official scripted training branch/loss."""
import argparse,collections,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from common import R,M3,config,read,sha,atomic,gpu_guard

def upstream(name):
 import ast
 path=R/'assets/finetune_upstream.py'
 tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 ns={'torch':torch,'optim':torch.optim}
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
 return ns[name]

class Samples:
 def __init__(self):
  import zarr
  self.c=config();self.rows=[r for r in read(M3/'MV-Generalist.json')['entries'] if r['supervised_role']=='train']
  assert len(self.rows)==156
  self.sources=sorted({r['source_id'] for r in self.rows});self.handles={};self.groups={}
  for r in self.rows:
   j=read(M3/'cache'/r['id']/'CACHE.json');assert j['manifest_sha256']==self.c['manifest_sha256']
   assert j['valid_path'] is None,'Do not silently treat ignored pixels as background'
   key=tuple(slice(a,b) for a,b in j['offsets'])
   im=np.load(j['paths']['image'],mmap_mode='r')[key];lab=np.load(j['paths']['label'],mmap_mode='r')[key]
   self.handles[r['id']]=(im,lab,j);self.groups[r['id']]=zarr.open_group(r['image_path'],mode='r')
 def one(self,index,forced_source=None):
  from empanada.data.utils import heatmap_and_offsets
  from skimage.measure import label
  rng=np.random.default_rng(np.random.SeedSequence([self.c['seed'],index]))
  source=forced_source or self.sources[int(rng.integers(len(self.sources)))];rows=[r for r in self.rows if r['source_id']==source]
  weights=[len(self.handles[r['id']][2]['valid_z']) for r in rows];idx=int(rng.integers(sum(weights)));k=int(np.searchsorted(np.cumsum(weights),idx,side='right'));r=rows[k]
  im,lab,j=self.handles[r['id']];z=j['valid_z'][idx-sum(weights[:k])];b=lab[z];h,w=b.shape
  fg=bool(rng.random()<.5);ys=xs=None
  if fg:ys,xs=np.nonzero(b>0)
  if ys is not None and len(ys):
   i=int(rng.integers(len(ys)));y=int(np.clip(ys[i]-128,0,max(0,h-256)));x=int(np.clip(xs[i]-128,0,max(0,w-256)))
  else:fg=False;y=int(rng.integers(max(1,h-256+1)));x=int(rng.integers(max(1,w-256+1)))
  a=np.array(im[z,y:y+256,x:x+256],np.float32,copy=True)
  zz=z+r['region_zyx'][0][0];yy=y+r['region_zyx'][1][0];xx=x+r['region_zyx'][2][0]
  ids=np.asarray(self.groups[r['id']]['mito'][zz,yy:yy+min(256,h-y),xx:xx+min(256,w-x)])
  assert np.array_equal(ids>0,b[y:y+256,x:x+256]>0),'Instance labels must exactly match existing semantic GT'
  # Binary-only annotations require explicit 2D connected-component pseudo-instances.
  # Multi-ID annotations retain their IDs; relabel densely only to bound regionprops allocation.
  values,inv=np.unique(ids,return_inverse=True)
  if np.count_nonzero(values)>1:
   ids=(inv.reshape(ids.shape)+(values[0]!=0)).astype(np.int32);mode='original_instance_ids'
  else:ids=label(ids>0,connectivity=1).astype(np.int32);mode='binary_2d_cc_pseudo_instances'
  flips=[bool(rng.random()<.5),bool(rng.random()<.5)]
  for axis,flip in enumerate(flips):
   if flip:a=np.flip(a,axis);ids=np.flip(ids,axis)
  pad=((0,256-a.shape[0]),(0,256-a.shape[1]));valid=np.pad(np.ones(a.shape,bool),pad,constant_values=False)
  a=np.pad(a,pad,constant_values=127.5);ids=np.pad(ids,pad,constant_values=0).copy();heat,offset=heatmap_and_offsets(ids,6)
  imtensor=torch.from_numpy(((a/255-self.c['normalization']['mean'])/self.c['normalization']['std']).copy())[None]
  target={'sem':torch.from_numpy((ids>0).astype(np.float32)),'ctr_hmp':torch.from_numpy(heat),'offsets':torch.from_numpy(offset),'valid':torch.from_numpy(valid)}
  coords={'source':source,'region':r['id'],'z_native':zz,'y_native':yy,'x_native':xx,'flips':flips,'foreground_centered':fg,'target_mode':mode}
  return imtensor,target,coords
 def batch(self,step,forced_source=None):
  batch=[self.one(step*self.c['batch_size']+i,forced_source) for i in range(self.c['batch_size'])]
  return torch.stack([b[0] for b in batch]).cuda(),{k:torch.stack([b[1][k] for b in batch]).cuda() for k in batch[0][1]},[b[2] for b in batch]

class StepBatches(torch.utils.data.Dataset):
 def __init__(self):self.samples=None
 def __len__(self):return config()['optimizer_steps']
 def __getitem__(self,step):
  if self.samples is None:self.samples=Samples()
  batch=[self.samples.one(step*config()['batch_size']+i) for i in range(config()['batch_size'])]
  return torch.stack([b[0] for b in batch]),{k:torch.stack([b[1][k] for b in batch]) for k in batch[0][1]},[b[2] for b in batch]

def panoptic_loss(criterion,output,target):
 """Official loss exactly on unpadded batches; exclude synthetic padding otherwise."""
 from empanada.models.point_rend import point_sample
 v=target['valid']
 if v.all():return criterion(output,target)
 sem=target['sem'];n=int(v.sum());assert n>0
 ce=criterion.ce_loss(output['sem_logits'][:,0][v].reshape(1,1,1,n),sem[v].reshape(1,1,n))
 mse=criterion.mse_loss(output['ctr_hmp'][:,0][v],target['ctr_hmp'][:,0][v])
 l1=criterion.l1_loss(output['offsets'],target['offsets'],((sem>0)&v).unsqueeze(1))
 pv=point_sample(v.unsqueeze(1).float(),output['point_coords'],mode='nearest',align_corners=False)>0
 pl=point_sample(sem.unsqueeze(1),output['point_coords'],mode='nearest',align_corners=False)
 pr=torch.nn.functional.binary_cross_entropy_with_logits(output['sem_points'][pv],pl[pv]) if pv.any() else output['sem_points'].sum()*0
 total=criterion.ce_weight*ce+criterion.mse_weight*mse+criterion.l1_weight*l1+criterion.pr_weight*pr
 return total,{'ce':float(ce.detach()),'l1':float(l1.detach()),'mse':float(mse.detach()),'pointrend_ce':float(pr.detach()),'total_loss':float(total.detach())}

def run(smoke=False):
 from empanada.losses import PanopticLoss
 gpu=gpu_guard();c=config();torch.set_num_threads(2);random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
 assert sha(R/'assets/finetune_upstream.py')==c['official_finetune_sha256']
 assert sha(Path(c['initial_checkpoint']))==c['initial_checkpoint_sha256']
 out=R/('smoke' if smoke else 'training');out.mkdir(parents=True,exist_ok=True)
 model=torch.jit.load(c['initial_checkpoint'],map_location='cpu').cuda()
 for p in model.parameters():p.requires_grad_(True)
 model.train();criterion=PanopticLoss(**c['criterion_params']).cuda()
 optimizer=upstream('configure_optimizer')(model,'AdamW',**c['optimizer_params'])
 scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,total_steps=c['optimizer_steps'],**c['schedule_params'])
 scaler=torch.amp.GradScaler('cuda',init_scale=1024);samples=Samples() if smoke else None;history=hashlib.sha256();counts=collections.Counter();modes=collections.Counter();start=time.time()
 loader=None if smoke else iter(torch.utils.data.DataLoader(StepBatches(),batch_size=None,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn'))
 steps=12 if smoke else c['optimizer_steps'];losses=[];curve=(out/'training_curve.jsonl').open('w')
 for step in range(steps):
  if smoke:images,target,coords=samples.batch(step,samples.sources[step])
  else:
   images,target,coords=next(loader);images=images.cuda(non_blocking=True);target={k:v.cuda(non_blocking=True) for k,v in target.items()}
  for attempt in range(8):
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):
    output=model(images);loss,aux=panoptic_loss(criterion,output,target)
   assert torch.isfinite(loss),f'Nonfinite loss step {step}'
   scaler.scale(loss).backward();scaler.unscale_(optimizer)
   grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'])
   old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
   if scaler.get_scale()>=old_scale:
    assert torch.isfinite(grad);break
   print('AMP_OVERFLOW_RETRY_SAME_BATCH',step+1,attempt+1,scaler.get_scale(),flush=True)
  else:raise RuntimeError('Persistent nonfinite gradients; no optimizer step counted')
  scheduler.step();history.update(json.dumps(coords,sort_keys=True).encode())
  counts.update(p['source'] for p in coords);modes.update(p['target_mode'] for p in coords);losses.append(aux)
  entry={'step':step+1,'seconds':time.time()-start,'lr':scheduler.get_last_lr()[0],'gradient_norm':float(grad),'amp_retry_count':attempt,'amp_scale':scaler.get_scale(),**aux}
  curve.write(json.dumps(entry)+'\n')
  if smoke or (step+1)%100==0:
   curve.flush();atomic(out/'progress.json',{'status':'running','step':step+1,'optimizer_steps':steps,'physical_gpu':gpu,'gpu0_used':False,'sampling_sha256':history.hexdigest(),'sample_source_counts':dict(counts),'target_mode_counts':dict(modes),**entry});print('MV5_TRAIN',step+1,aux,'seconds',entry['seconds'],flush=True)
  if not smoke and step+1 in [10000,20000,40000]:save(model,optimizer,scheduler,scaler,out,step+1,c,history.hexdigest())
 curve.close()
 if smoke:
  save(model,optimizer,scheduler,scaler,out,steps,c,history.hexdigest())
  model.eval();x=images[:1]
  with torch.no_grad():before=model(x)['sem_logits'].float()
  clone=torch.jit.load(str(out/f'model_step{steps:06d}.pth'),map_location='cuda').eval()
  with torch.no_grad():after=clone(x)['sem_logits'].float()
  assert torch.equal(before,after)
  trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);assert trainable==55158486
  atomic(out/'smoke_complete.json',{'status':'PASS','physical_gpu':gpu,'steps':steps,'sources_covered':sorted(counts),'trainable_parameters':trainable,'all_official_loss_terms_finite':True,'checkpoint_inference_exact':True,'han_annotation_mask_read':False,'target_mode_counts':dict(modes),'gradient_norm_last':float(grad)})
 else:atomic(out/'training_complete.json',{'status':'complete','completed_steps':steps,'physical_gpu':gpu,'gpu0_used':False,'sampling_sha256':history.hexdigest(),'sample_source_counts':dict(counts),'target_mode_counts':dict(modes),'seconds':time.time()-start,'protocol_sha256':sha(R/'MV5_protocol.json')})

def save(model,optimizer,scheduler,scaler,out,step,c,sampling):
 p=out/f'model_step{step:06d}.pth';model.save(str(p)+'.tmp');Path(str(p)+'.tmp').replace(p)
 resume=out/f'resume_step{step:06d}.pth';torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'protocol':c},resume)
 atomic(p.with_suffix('.json'),{'path':str(p),'sha256':sha(p),'step':step,'sampling_sha256':sampling,'protocol_sha256':sha(R/'MV5_protocol.json'),'initial_checkpoint_sha256':c['initial_checkpoint_sha256'],'resume_path':str(resume),'resume_sha256':sha(resume)})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');run(p.parse_args().smoke)
