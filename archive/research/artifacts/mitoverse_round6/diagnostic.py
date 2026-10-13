"""Fixed train-only eight-patch diagnostic. Checkpoints cannot be used for formal probes."""
import os,sys,json,time,csv,hashlib,argparse,math
from pathlib import Path
from settings import *
sys.path.insert(0,str(R4))
import common as c
import train as old
import torch,numpy as np,zarr
from PIL import Image,ImageDraw
def sd_hash(e,exclude_last4=False):
 h=hashlib.sha256()
 for k,v in e.state_dict().items():
  if exclude_last4 and any(k.startswith(f'blocks.{i}.') for i in range(20,24)):continue
  h.update(k.encode());h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def features(net,x):
 with torch.autocast('cuda',dtype=torch.bfloat16):return net.dino_encoder.get_intermediate_layers(x,n=net.interaction_indices,reshape=True)
def head_forward(net,fs):return net.up_projection(torch.cat([g(f.float()) for g,f in zip(net.last_gates,fs)],dim=1))
def fw(net,x):return head_forward(net,features(net,x))
def select():
 target=BASE/'fixed_patches.npz'
 if target.exists():return np.load(target),json.loads((BASE/'fixed_patches.json').read_text())
 split=json.loads((R4/'split_mitoverse_probe_v1.json').read_text());fg=json.loads((R4/'train_foreground_coordinates.json').read_text());rng=np.random.default_rng(6017)
 images=[];labels=[];records=[];handles={}
 def patch(vid,z,y,x,kind):
  r=split['volumes'][vid];assert r['split']=='train';p=Path(r['image_path']).parent
  assert not (p/'mask/.zarray').exists(), 'Ambiguous mask excluded from diagnostic only'
  if vid not in handles:handles.clear();handles[vid]=(zarr.open_array(r['image_path'],mode='r'),zarr.open_array(r['label_path'],mode='r'))
  a,b=handles[vid];im=np.asarray(a[z,y:y+256,x:x+256]);lab=np.asarray(b[z,y:y+256,x:x+256])>0
  if im.shape!=(256,256):return False
  if bool(lab.any())!=(kind=='foreground'):return False
  images.append(im);labels.append(lab.astype(np.int64));records.append(dict(volume_id=vid,source=r['source'],split='train',z=int(z),y=int(y),x=int(x),kind=kind,foreground=int(lab.sum()),voxel_nm_zyx=r['voxel_nm_z_y_x'],shape=[256,256],normalization='N2',augmentation='none'));return True
 for source in ['muller21','kunduri22']:
  vids=sorted(v for v in split['train'] if split['volumes'][v]['source']==source)
  got=0
  for vid in vids:
   d,h,w=json.loads(split['volumes'][vid]['shape_z_y_x'])
   for z,cy,cx in fg[vid][::max(1,len(fg[vid])//12)]:
    if patch(vid,z,min(max(cy-128,0),h-256),min(max(cx-128,0),w-256),'foreground'):got+=1;break
   if got==3:break
  assert got==3, 'Need three foreground patches per selected source'
 # Background must be actually empty in a fully annotated training array; no relabeling.
 candidates=sorted(v for v in split['train'] if split['volumes'][v]['source'] in ['muller21','kunduri22','turner22'])
 for vid in candidates:
  d,h,w=json.loads(split['volumes'][vid]['shape_z_y_x'])
  if min(h,w)<256:continue
  for z in sorted(set([0,d-1,d//2])):
   for y,x in [(0,0),(h-256,w-256),(0,w-256),(h-256,0)]:
    if patch(vid,z,y,x,'background') and len(images)==8:break
   if len(images)==8:break
  if len(images)==8:break
 assert len(images)==8,'Insufficient background patches in bounded train-only selection'
 np.savez_compressed(target,images=np.stack(images),labels=np.stack(labels));c.save(BASE/'fixed_patches.json',records)
 return np.load(target),records
def visualization(raw,y,p,path):
 canvas=Image.new('RGB',(256*3,8*280),'white');draw=ImageDraw.Draw(canvas)
 for i in range(8):
  for j,a in enumerate([raw[i],y[i]*255,p[i]*255]):canvas.paste(Image.fromarray(a.astype(np.uint8)).convert('RGB'),(256*j,280*i+24))
  draw.text((4,280*i+4),f'{i}: image | label | prediction',fill='black')
 canvas.save(path)
def inference(net,raw):
 net.eval();x=torch.from_numpy(np.stack([c.normalize(raw,'N2')])).cuda()
 with torch.inference_mode():
  a=c.forward(net,x).float().softmax(1)[:,1].cpu().numpy()[0]
  b=c.forward(net,x).float().softmax(1)[:,1].cpu().numpy()[0]
  stitched=old.slide(net,raw,'N2')
 repeat=float(np.max(np.abs(a-b)));tol=max(1e-5,2*repeat+1e-6);single=float(np.max(np.abs(a-stitched)))
 assert single<=tol
 # Overlap test with identical real windows and an independent implementation of fusion.
 mosaic=np.tile(raw,(2,2))[:384,:384];coords=[(y,x) for y in old.starts(384) for x in old.starts(384)]
 xx=torch.from_numpy(np.stack([c.normalize(mosaic[y:y+256,x:x+256],'N2') for y,x in coords])).cuda()
 with torch.inference_mode():probs=c.forward(net,xx).float().softmax(1)[:,1].cpu().numpy()
 g=np.zeros((256,256),np.float32);g[128,128]=1;g=old.gaussian_filter(g,32);g=np.maximum(g/g.max(),1e-7)
 numerator=np.zeros((384,384),np.float64);denominator=np.zeros_like(numerator)
 for (oy,ox),p in zip(coords,probs):numerator[oy:oy+256,ox:ox+256]+=p*g;denominator[oy:oy+256,ox:ox+256]+=g
 q=old.slide(net,mosaic,'N2');overlap=float(np.max(np.abs(q-numerator/denominator)));assert overlap<=tol and np.min(denominator)>0
 # FP32 identical-context window vs production sliding path; retain model weights unchanged.
 def fp32_forward(net,x):
  with torch.no_grad():return head_forward(net,net.dino_encoder.get_intermediate_layers(x,n=net.interaction_indices,reshape=True))
 with torch.inference_mode():pa=fp32_forward(net,x).softmax(1)[:,1].cpu().numpy()[0]
 previous=old.forward;old.forward=fp32_forward
 try:pb=old.slide(net,raw,'N2')
 finally:old.forward=previous
 error=float(np.max(np.abs(pa-pb)));assert error<=1e-5
 r=dict(status='pass',bf16_repeat_max_abs=repeat,bf16_locked_tolerance=tol,single_window_sliding_max_abs=single,overlap_fusion_max_abs=overlap,min_weight=float(denominator.min()),fp32_max_abs=error,threshold=.5,softmax_foreground_channel=1,original_grid=True,padding='reflect image; -1 labels',interpolation='none; no label resizing',n2_range=[float(x.min()),float(x.max())],same_context=True)
 c.save(BASE/'inference_consistency.json',r)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--x1',action='store_true');a=ap.parse_args();out=Path(os.environ['RUN_OUTPUT']);c.seed(0)
 ds,records=select();raw=ds['images'];yy=ds['labels'];xs=torch.from_numpy(np.stack([c.normalize(r,'N2') for r in raw])).cuda();ys=torch.from_numpy(yy).cuda()
 net=c.model('P1',0);net.eval();before=sd_hash(net.dino_encoder,exclude_last4=a.x1)
 if not a.x1:inference(net,raw[0])
 hp=[p for p in net.parameters() if p.requires_grad];ep=[];initial={}
 if a.x1:
  for block in net.dino_encoder.blocks[20:24]:block.requires_grad_(True);ep.extend(block.parameters())
  initial={k:p.detach().cpu().clone() for k,p in net.dino_encoder.named_parameters() if p.requires_grad}
 params=hp+ep;frozen=[p for p in net.dino_encoder.parameters() if not p.requires_grad]
 opt=torch.optim.AdamW([dict(params=hp,lr=1e-4,base_lr=1e-4)]+([dict(params=ep,lr=3e-6,base_lr=3e-6)] if ep else []),betas=(.9,.98),weight_decay=.05)
 c.save(out/'parameter_audit.json',dict(x1=a.x1,encoder_training=net.dino_encoder.training,head_parameter_count=sum(p.numel() for p in hp),last4_parameter_count=sum(p.numel() for p in ep),trainable_encoder_names=list(initial),frozen_parameters=sum(p.numel() for p in frozen),optimizer=[dict(lr=g['lr'],parameter_count=sum(p.numel() for p in g['params'])) for g in opt.param_groups],forward_microbatch=2,effective_batch=8,backward_calls_per_step=1,steps=1000,schedule='R5 first 1000 of 10k schedule',diagnostic_only=True))
 cached=None
 if not a.x1:
  # Encoder is deterministic/eval and frozen. Cache only these eight patches, not a dataset.
  with torch.no_grad():
   chunks=[features(net,xs[i:i+2]) for i in range(0,8,2)];cached=[torch.cat([f[k] for f in chunks]) for k in range(4)]
   ref=c.forward(net,xs[:2]);check=head_forward(net,[f[:2] for f in cached]);assert torch.equal(ref,check)
  del chunks,ref,check
 curves=[];start=time.time();pred=None
 def evaluate(step,loss):
  net.eval()
  with torch.inference_mode():
   logits=torch.cat([fw(net,xs[i:i+2]) if cached is None else head_forward(net,[f[i:i+2] for f in cached]) for i in range(0,8,2)])
   p=logits.softmax(1)[:,1]>=.5;d=[c.metrics(c.confusion(p[i].cpu().numpy(),yy[i]>0))['dice'] for i in range(6)]
  row=dict(step=step,foreground_mean_dice=float(np.mean(d)),loss=loss,seconds=time.time()-start,head_lr=opt.param_groups[0]['lr'],last4_lr=opt.param_groups[-1]['lr'] if a.x1 else 0)
  curves.append(row)
  with (out/'overfit_curves.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerows(curves)
  c.save(out/'progress.json',row);print(json.dumps(row),flush=True);return p.cpu().numpy()
 pred=evaluate(0,None)
 visualization(raw,yy,pred,out/'fixed_before.png')
 for step in range(1,1001):
  net.train();net.dino_encoder.eval();scale=min(1.,step/50)*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/10000)))
  for g in opt.param_groups:g['lr']=g['base_lr']*scale
  opt.zero_grad(set_to_none=True)
  # Build one global-batch loss: Dice must not be averaged across microbatches.
  logits=torch.cat([fw(net,xs[i:i+2]) if cached is None else head_forward(net,[f[i:i+2] for f in cached]) for i in range(0,8,2)])
  loss=old.loss(logits,ys);assert torch.isfinite(loss);loss.backward()
  assert all(p.grad is None for p in frozen)
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
  torch.nn.utils.clip_grad_norm_(params,12,error_if_nonfinite=True);opt.step()
  if step%100==0:pred=evaluate(step,float(loss.detach()))
 after=sd_hash(net.dino_encoder,exclude_last4=a.x1);assert before==after,'Frozen encoder changed'
 updates={k:float(torch.linalg.vector_norm(p.detach().cpu()-initial[k])/(torch.linalg.vector_norm(initial[k])+1e-12)) for k,p in net.dino_encoder.named_parameters() if k in initial}
 layers={str(i):[v for k,v in updates.items() if k.startswith(f'blocks.{i}.')] for i in range(20,24)} if a.x1 else {}
 if a.x1:assert all(any(v>0 and np.isfinite(v) for v in vv) for vv in layers.values())
 visualization(raw,yy,pred,out/'fixed_after.png');c.save(out/'relative_updates.json',updates)
 torch.save(dict(head=c.head(net),encoder_last4={k:p.detach().cpu() for k,p in net.dino_encoder.named_parameters() if k in initial},diagnostic_only=True,forbidden_for_formal_initialization=True),out/'diagnostic_only.pth')
 c.save(out/'result.json',dict(status='complete',diagnostic_only=True,steps=1000,samples=8000,unique_patches=8,seconds=time.time()-start,foreground_mean_dice=curves[-1]['foreground_mean_dice'],engineering_gate=curves[-1]['foreground_mean_dice']>=.90,gradient_finite=True,frozen_weights_unchanged=True,last4_relative_updates=updates,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),mask_policy='unmasked training sources only; ambiguous han24 excluded only from diagnostic'))
if __name__=='__main__':main()
