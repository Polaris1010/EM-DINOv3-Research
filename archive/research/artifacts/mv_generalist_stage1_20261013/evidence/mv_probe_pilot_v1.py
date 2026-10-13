"""Official Primus frozen probe for locked MitoVerse source-volume DEVELOPMENT."""
from pathlib import Path
import sys,os,json,random,importlib.util,gc,time,argparse,math,collections,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mv_data import Samples,manifest,open_cache,sha,atomic
import numpy as np,torch
import torch.nn.functional as F
def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(2)
def state_hash(state):
 h=hashlib.sha256()
 for n,t in sorted(state.items()):
  h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
  h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def head_state(net):return {k:v.detach().cpu() for k,v in net.state_dict().items() if not k.startswith('dino_encoder.')}
def build(model_id,s):
 specs=json.loads((R/'model_registry.json').read_text())['models'];m=next(x for x in specs if x['id']==model_id)
 for p,h in [(m['checkpoint_path'],m['sha256']),(m['resolved_config_path'],m['resolved_config_sha256']),(m['primus_source_path'],m['primus_source_sha256'])]:assert sha(p)==h
 sys.path[:0]=[m['dinov3_root'],m['nnunet_root']]
 from omegaconf import OmegaConf
 from dinov3.models import build_model_from_cfg
 enc,dim=build_model_from_cfg(OmegaConf.load(m['resolved_config_path']),only_teacher=True)
 assert dim==1024 and enc.n_storage_tokens==4 and enc.patch_size==16
 saved=torch.load(m['checkpoint_path'],map_location='cpu',mmap=True,weights_only=True)
 if model_id=='P0':sd=saved
 else:
  assert saved['step']=={'B100k':100000,'B150k':150000}[model_id] and saved['manifest_sha256']==m['ssl_manifest_sha256'];sd=saved['backbone']
 assert len(sd)==368;enc.load_state_dict(sd,strict=True,assign=True);enc.requires_grad_(False).eval()
 spec=importlib.util.spec_from_file_location('mv_official_primus',R/'src/official_primus.py')
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 seed(s);net=mod.Primus_Multiscale(embed_dim=1024,patch_embed_size=16,num_classes=2,dino_encoder=enc,interaction_indices=[5,11,17,23])
 assert not any('gate' in k.lower() for k,_ in net.named_parameters()) and not hasattr(net,'last_gates')
 assert sum(p.numel() for p in net.parameters() if p.requires_grad)==44576514
 net.cuda();del saved,sd;gc.collect();return net,m
def forward(net,x):
 net.dino_encoder.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  f=net.dino_encoder.get_intermediate_layers(x.repeat(1,3,1,1),n=net.interaction_indices,reshape=True)
 return net.up_projection(torch.cat([v.float() for v in f],1))
def loss(logits,y):
 valid=y>=0;t=(y==1).float();p=logits.float().softmax(1)[:,1]
 assert valid.any()
 return F.cross_entropy(logits.float(),y,ignore_index=-1)+1-(2*(p*t*valid).sum()+1e-5)/((p*valid).sum()+(t*valid).sum()+1e-5)
def confusion(pred,gt):
 return [int(np.logical_and(pred,gt).sum()),int(np.logical_and(pred,~gt).sum()),int(np.logical_and(~pred,gt).sum()),int(np.logical_and(~pred,~gt).sum())]
def overlap(c):
 tp,fp,fn,tn=map(int,c);den=2*tp+fp+fn
 return {'dice':2*tp/den if den else 1.,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
 'tp':tp,'fp':fp,'fn':fn,'tn':tn,'foreground_empty_prediction':tp+fp==0}
def starts(n):return sorted(set(range(0,max(1,n-256+1),128))|{max(0,n-256)})
@torch.no_grad()
def sliding(net,img):
 H,W=img.shape;acc=np.zeros((H,W),np.float32);weight=np.zeros((H,W),np.uint16)
 for y in starts(H):
  for x in starts(W):
   a=np.asarray(img[y:y+256,x:x+256],np.float32);h,w=a.shape
   a=np.pad(a,((0,256-h),(0,256-w)),constant_values=127.5)
   z=torch.from_numpy(((a/255-.5)/.25).copy())[None,None].cuda()
   p=forward(net,z).softmax(1)[0,1,:h,:w].cpu().numpy();assert np.isfinite(p).all()
   acc[y:y+h,x:x+w]+=p;weight[y:y+h,x:x+w]+=1
 assert (weight>0).all();return acc/weight>.5
@torch.no_grad()
def validate(net,out,full=False):
 net.eval();rows=[];t0=time.time()
 for row in manifest()['entries']:
  if row['supervised_role']!='dev':continue
  im,lab=open_cache(row);Z=im.shape[0];slices=list(range(Z)) if full else sorted(set(np.linspace(0,Z-1,min(8,Z),dtype=int)))
  pred=np.lib.format.open_memmap(out/(row['id']+'_prediction.npy'),mode='w+',shape=im.shape,dtype=np.uint8) if full else None
  c=np.zeros(4,np.int64)
  for k,z in enumerate(slices):
   p=sliding(net,im[z]);c+=confusion(p,lab[z].astype(bool))
   if pred is not None:pred[z]=p
   if full and k%50==0:atomic(out/'evaluation_progress.json',{'volume':row['id'],'z_completed':k+1,'z_total':len(slices),'elapsed_s':time.time()-t0})
  if pred is not None:pred.flush();del pred
  rows.append({'volume_id':row['id'],'source_id':row['source_id'],'slice_count':len(slices),'spacing_zyx_nm':row['spacing_zyx_nm'],**overlap(c)})
 bysource=collections.defaultdict(list)
 for r in rows:bysource[r['source_id']].append(r['dice'])
 means={k:float(np.mean(v)) for k,v in bysource.items()}
 return {'rows':rows,'descriptive_all_source_macro_dice':float(np.mean(list(means.values()))),'source_means':means,
  'primary_eligible_sources':[k for k,v in bysource.items() if len(v)>=2],'full_volume':full,'test_labels_accessed':False}
def smoke(model_id,s,out):
 net,m=build(model_id,s);initial=state_hash(head_state(net));before=state_hash(net.dino_encoder.state_dict());samples=Samples(s)
 pars=[p for p in net.parameters() if p.requires_grad];opt=torch.optim.AdamW(pars,lr=1e-4);rows=[]
 for i,source in enumerate(samples.sources):
  x,y,coords=samples.sample(i,forced_source=source);opt.zero_grad(set_to_none=True);v=loss(forward(net,x),y)
  assert torch.isfinite(v);v.backward();assert all(p.grad is None for p in net.dino_encoder.parameters())
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pars);opt.step()
  rows.append({'source':source,'loss':float(v.detach()),'sample':coords})
 assert before==state_hash(net.dino_encoder.state_dict())
 # Held-out real data, one 256 ROI each source, pipeline smoke only.
 dev=[]
 for r in manifest()['entries']:
  if r['supervised_role']=='dev':
   im,gt=open_cache(r);a=np.array(im[len(im)//2,:256,:256],copy=True);p=sliding(net,a)
   dev.append({'volume':r['id'],**overlap(confusion(p,gt[len(gt)//2,:256,:256].astype(bool)))})
 # Verify both samplers actually honor source probabilities and train-only membership.
 from collections import Counter
 n=20000;sampling={}
 for mode in ['slice_uniform','source_balanced']:
  sampler=Samples(s,mode);counts=Counter()
  for k in range(n):
   rng=np.random.default_rng(np.random.SeedSequence([s,k]));selected=sampler.rows
   if mode=='source_balanced':source=sampler.sources[int(rng.integers(len(sampler.sources)))];selected=[r for r in selected if r['source_id']==source]
   sizes=np.array([r['shape_zyx'][0] for r in selected]);j=int(np.searchsorted(np.cumsum(sizes),int(rng.integers(sizes.sum())),side='right'));counts[selected[j]['source_id']]+=1
  sampling[mode]=dict(counts)
 atomic(out/'smoke.json',{'status':'PASS','model':model_id,'head_initial_sha256':initial,'encoder_unchanged':True,
  'trainable_parameters':44576514,'train':rows,'dev_ROI_smoke':dev,'sampling_draws':sampling,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
  'manifest_sha256':sha(R/'MV-Dev-v1.json'),'test_labels_accessed':False})
 print('SMOKE_PASS',model_id,initial,flush=True)
def train(model_id,s,steps,out,pilot=False):
 net,m=build(model_id,s);initial=state_hash(head_state(net));samples=Samples(s)
 pars=[p for p in net.parameters() if p.requires_grad];opt=torch.optim.AdamW(pars,lr=1e-4,weight_decay=.05,betas=(.9,.98))
 protocol=json.loads((R/'MV_frozen_probe_protocol.yaml').read_text());mhash=sha(R/'MV-Dev-v1.json');phash=sha(R/'MV_frozen_probe_protocol.yaml')
 meta={'model':model_id,'seed':s,'steps':steps,'pilot':pilot,'decoder_initial_sha256':initial,'manifest_sha256':mhash,
  'protocol_sha256':phash,'encoder_sha256':m['sha256'],'encoder_frozen':True,'gates':False,'test_labels_accessed':False}
 atomic(out/'run_protocol.json',meta);t0=time.time();curve=[];recent=[];counts=collections.Counter();samples_h=hashlib.sha256()
 for i in range(steps):
  net.train();net.dino_encoder.eval();x,y,coord=samples.sample(i);counts[coord['source']]+=1
  samples_h.update(json.dumps(coord,sort_keys=True).encode());lr=1e-4*min(1.,(i+1)/100)
  # Pilot uses fixed LR after warmup; formal common-budget run uses common cosine.
  if not pilot:lr*=.1+.9*.5*(1+np.cos(np.pi*i/steps))
  for g in opt.param_groups:g['lr']=lr
  opt.zero_grad(set_to_none=True);v=loss(forward(net,x),y);assert torch.isfinite(v);v.backward()
  assert all(p.grad is None for p in net.dino_encoder.parameters());torch.nn.utils.clip_grad_norm_(pars,12,error_if_nonfinite=True);opt.step()
  recent.append(float(v.detach()));recent=recent[-200:]
  if (i+1)%100==0:
   atomic(out/'progress.json',{'model':model_id,'seed':s,'step':i+1,'steps':steps,'loss_median200':float(np.median(recent)),'seconds':time.time()-t0,'source_draws':dict(counts)})
  if (i+1) in protocol['budget_pilot_checkpoints'] or i+1==steps:
   val=validate(net,out);entry={'step':i+1,'loss_median200':float(np.median(recent)),**val};curve.append(entry);atomic(out/'curve.json',curve)
   print('CHECKPOINT',model_id,s,i+1,entry['descriptive_all_source_macro_dice'],flush=True)
   if pilot:
    torch.save({'head':head_state(net),'step':i+1,'metadata':meta},out/f'pilot_head_step{i+1:06d}.pth')
    if len(curve)>=3:
     last=curve[-3:];improvements=[last[k+1]['descriptive_all_source_macro_dice']-last[k]['descriptive_all_source_macro_dice'] for k in range(2)]
     losses=[(last[k]['loss_median200']-last[k+1]['loss_median200'])/max(last[k]['loss_median200'],1e-8) for k in range(2)]
     # Negative metric movement qualifies as no extra benefit, but a large loss increase indicates instability.
     if all(d<=.005 for d in improvements) and all(-.05<=d<=.05 for d in losses):break
 meta.update({'completed_steps':i+1,'samples_sequence_sha256':samples_h.hexdigest(),'source_draws':dict(counts),'seconds':time.time()-t0})
 torch.save({'head':head_state(net),'step':i+1,'metadata':meta},out/'head_last.pth')
 if not pilot:meta['validation']=validate(net,out,full=True)
 atomic(out/'metrics.json',{'status':'complete',**meta});print('TRAIN_COMPLETE',model_id,s,i+1,flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['P0','B100k','B150k'],required=True);ap.add_argument('--seed',type=int,default=20261013)
 ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--pilot',action='store_true');ap.add_argument('--steps',type=int)
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if a.smoke:smoke(a.model,a.seed,a.out)
 else:train(a.model,a.seed,a.steps,a.out,a.pilot)
if __name__=='__main__':main()
