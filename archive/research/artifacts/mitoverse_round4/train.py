from common import *
from data import loader
import argparse,time,signal
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
signal.signal(signal.SIGINT,signal.default_int_handler)

def loss(logits,y):
 valid=y>=0;t=(y==1).float();p=logits.float().softmax(1)[:,1]
 assert valid.any();return F.cross_entropy(logits.float(),y,ignore_index=-1)+1-(2*(p*t*valid).sum()+1e-5)/((p*valid).sum()+(t*valid).sum()+1e-5)
def starts(n):
 count=int(np.ceil(max(0,n-256)/128))+1
 return [round(i*(n-256)/(count-1)) for i in range(count)] if count>1 else [0]
@torch.inference_mode()
def slide(net,raw,norm):
 h,w=raw.shape;im=np.pad(raw,((0,max(0,256-h)),(0,max(0,256-w))),mode='reflect');H,W=im.shape
 g=np.zeros((256,256),np.float32);g[128,128]=1;g=gaussian_filter(g,32);g/=g.max();g=np.maximum(g,1e-7)
 pred=np.zeros((H,W),np.float32);weight=np.zeros((H,W),np.float32);coords=[(y,x) for y in starts(H) for x in starts(W)]
 for i in range(0,len(coords),8):
  batch=coords[i:i+8];a=np.stack([normalize(im[y:y+256,x:x+256],norm) for y,x in batch]);q=forward(net,torch.from_numpy(a).cuda()).float().softmax(1)[:,1].cpu().numpy();assert np.isfinite(q).all()
  for (y,x),p in zip(batch,q):pred[y:y+256,x:x+256]+=p*g;weight[y:y+256,x:x+256]+=g
 assert (weight>0).all();return (pred/weight)[:h,:w]

def evaluate(net,norm,out,splits,full=False):
 net.eval();split=json.loads((BASE/'split_mitoverse_probe_v1.json').read_text());result={};t0=time.time()
 for sp in splits:
  for vid in split[sp]:
   r=split['volumes'][vid];raw=zarr.open_array(r['image_path'],mode='r');lab=zarr.open_array(r['label_path'],mode='r');assert raw.shape==lab.shape
   zs=range(raw.shape[0]) if full else sorted(set(np.linspace(0,raw.shape[0]-1,min(8,raw.shape[0]),dtype=int)))
   predstore=None
   if full:
    gp=zarr.open_group(str(out/'predictions'/f'{vid}.zarr'),mode='w');predstore=gp.create_dataset('probability',shape=raw.shape,chunks=(1,256,256),dtype='float16');binary=gp.create_dataset('binary',shape=raw.shape,chunks=(1,256,256),dtype='uint8');gp.attrs.update({'grid':'original ZYX','threshold':.5,'tta':False,'source':r['source'],'voxel_nm_zyx':r['voxel_nm_z_y_x']})
   c=np.zeros(4,np.int64);seams=[];vstart=time.time()
   for z in zs:
    im=np.asarray(raw[z]);t=np.asarray(lab[z])>0;pr=slide(net,im,norm);p=pr>=.5;c+=confusion(p,t)
    if full:
     predstore[z]=pr.astype(np.float16);binary[z]=p.astype(np.uint8)
     if z==raw.shape[0]//2:
      # Compare probability jumps at tile starts and typical adjacent-pixel jumps.
      jumps=[float(np.abs(pr[y]-pr[y-1]).mean()) for y in starts(max(256,pr.shape[0]))[1:] if 0<y<pr.shape[0]]
      seams={'tile_start_jump_mean':float(np.mean(jumps)) if jumps else None,'all_horizontal_jump_mean':float(np.abs(np.diff(pr,axis=0)).mean())}
      from PIL import Image
      boundary=t ^ __import__('scipy').ndimage.binary_erosion(t)
      rgb=np.repeat(im[:,:,None],3,axis=2);rgb[p]=(rgb[p].astype(np.float32)*.5+np.array([255,125,0])*.5).astype(np.uint8);rgb[boundary]=[0,255,0]
      (out/'previews').mkdir(exist_ok=True);Image.fromarray(rgb).save(out/'previews'/f'{vid}.png')
   result[vid]=dict(source=r['source'],split=sp,ssl_pretrain_seen=r['ssl_pretrain_seen'],p1_em_seen=r['p1_em_seen'],p2_extra_seen=r['p2_extra_seen'],**metrics(c),seconds=time.time()-vstart,slices=len(zs),seam_diagnostic=seams)
   if full:assert sum(c)==int(np.prod(raw.shape))
   print('VOLUME',vid,result[vid]['dice'],'full',full,flush=True)
   if full:save(out/'volume_metrics_partial.json',result)
 sources={s:float(np.mean([v['dice'] for v in result.values() if v['source']==s])) for s in sorted({v['source'] for v in result.values()})}
 return dict(volumes=result,source_dice=sources,source_macro_dice=float(np.mean(list(sources.values()))),full=full,seconds=time.time()-t0)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--id',choices=['P0','P1','P2'],required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--norm',choices=['N1','N2'],default='N1');ap.add_argument('--mode',choices=['smoke','pilot','train','evaluate'],required=True);ap.add_argument('--head-from',type=Path);a=ap.parse_args();out=Path(os.environ['RUN_OUTPUT']);seed(a.seed)
 net=model(a.id,a.seed);params=[p for p in net.parameters() if p.requires_grad];assert all(not p.requires_grad for p in net.dino_encoder.parameters())
 if a.mode=='evaluate':
  ck=torch.load(a.head_from/'head_best.pth',map_location='cpu',weights_only=False);assert ck['id']==a.id and ck['seed']==a.seed and ck['norm']==a.norm;load_head(net,ck['head']);result=evaluate(net,a.norm,out,['validation','test1','test2','test3'],True);save(out/'metrics.json',dict(status='complete',id=a.id,seed=a.seed,selected_step=ck['step'],norm=a.norm,**result));return
 steps={'smoke':100,'pilot':500,'train':10000}[a.mode];opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.05,betas=(.9,.98));data=loader(a.seed,steps*8,a.norm);curve=[];best=-1;t0=time.time();digest=hashlib.sha256();testx=None
 for step,(x,y,indices) in enumerate(data,1):
  assert indices.tolist()==list(range((step-1)*8,step*8));digest.update(indices.numpy().tobytes());net.train();net.dino_encoder.eval();x=x.cuda();y=y.cuda()
  # A common 10k scheduler even in pilot/smoke, so pilots are training prefixes.
  lr=1e-4*min(1.,step/50)*(.1+.9*.5*(1+np.cos(np.pi*(step-1)/10000)))
  for g in opt.param_groups:g['lr']=lr
  opt.zero_grad(set_to_none=True);q=forward(net,x);v=loss(q,y);assert torch.isfinite(v);v.backward()
  assert all(p.grad is None for p in net.dino_encoder.parameters())
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
  gn=torch.nn.utils.clip_grad_norm_(params,12,error_if_nonfinite=True);opt.step()
  if step%25==0:
   row=dict(step=step,loss=float(v.detach()),grad_norm=float(gn),lr=lr,seconds=time.time()-t0,peak_gpu_bytes=torch.cuda.max_memory_allocated());print(json.dumps(row),flush=True);save(out/'progress.json',row)
  if step%1000==0 or step==steps:
   ck=dict(head=head(net),optimizer=opt.state_dict(),step=step,id=a.id,seed=a.seed,norm=a.norm,head_init_sha256=sha(BASE/f'shared_head_init_seed{a.seed}.pth'),schedule_sha256=sha(BASE/f'train_patch_schedule_seed{a.seed}.jsonl'),consumed_indices_sha256=digest.hexdigest())
   torch.save(ck,out/'checkpoint_latest.pth')
   if a.mode=='train':torch.save({k:v for k,v in ck.items() if k!='optimizer'},out/f'head_step{step}.pth')
   if a.mode!='smoke':
    val=evaluate(net,a.norm,out,['validation']);curve.append(dict(step=step,**val));save(out/'curve.json',curve)
    if val['source_macro_dice']>best:best=val['source_macro_dice'];torch.save({k:v for k,v in ck.items() if k!='optimizer'},out/'head_best.pth')
   testx=x[:1].detach().clone()
 # DataLoader exhausted and workers shut down before reload comparison.
 del data;net.eval()
 with torch.inference_mode():before=forward(net,testx).cpu()
 ck=torch.load(out/'checkpoint_latest.pth',map_location='cpu',weights_only=False);load_head(net,ck['head'])
 with torch.inference_mode():after=forward(net,testx).cpu()
 assert torch.equal(before,after),'Reload fixed-probe output mismatch'
 save(out/'result.json',dict(status='complete',mode=a.mode,id=a.id,seed=a.seed,norm=a.norm,steps=steps,encoder_no_grad=True,head_grad_finite=True,reload_exact=True,head_init_sha256=ck['head_init_sha256'],schedule_sha256=ck['schedule_sha256'],consumed_indices_sha256=digest.hexdigest(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-t0,best_validation_dice=best))
 del net,opt,params,ck,x,y,q,v,testx,before,after;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
