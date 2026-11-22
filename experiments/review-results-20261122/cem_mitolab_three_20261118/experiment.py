from pathlib import Path
import csv, hashlib, json, os, sys
import numpy as np
import torch
from prepare import X, CACHE, ROOT
B=X.parent
MB=B/'mb1_official_benchmark_20261015'
sys.path[:0]=[str(B/'mv_generalist_stage1_20261013/src'),str(MB/'code')]
from common import gpu_guard, Planes, atomic, sha
import mv_probe as p
SEED=20261014
STEPS=40000
model=sys.argv[1]
gpu=gpu_guard()
assert gpu=={'P0':1,'B100':2,'Barrier100':3}[model]
out=X/model
out.mkdir(exist_ok=True)
manifest=json.loads((X/'manifest.json').read_text())
groups=manifest['groups']
val=manifest['validation']
train=sorted(set(groups)-set(val))
net,spec=p.build('B100k' if model=='B100' else 'P0',SEED)
encoder_sha=spec['sha256']
if model=='Barrier100':
    cfg=json.loads((B/'pilot_c100_decoder_seeds_20261102/protocol.json').read_text())['models']['C1_Barrier_100k']
    enc=ROOT/cfg['encoder_relative']
    encoder_sha=sha(enc)
    assert encoder_sha==cfg['encoder_sha256']
    state=torch.load(enc,weights_only=True,map_location='cpu',mmap=True)
    net.dino_encoder.load_state_dict(state['backbone'],strict=True)
    del state
init=p.state_hash(p.head_state(net))
assert init=='2ec39fb096d1a0eceb3124a9e08549d0c6ac82665dd71f8e2bc7b5a8d56f48ec'
meta=dict(model=model,seed=SEED,steps=STEPS,encoder_sha256=encoder_sha,initial_sha256=init,manifest_sha256=sha(X/'manifest.json'),source_sha256=sha(__file__),gpu=gpu)
atomic(out/'protocol.json',meta)
params=[v for v in net.parameters() if v.requires_grad]
opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.05,betas=(.9,.98))
running=bytes(32)
start=0
checkpoints=sorted(out.glob('head_*.pth'))
if checkpoints:
    s=torch.load(checkpoints[-1],map_location='cpu',weights_only=True)
    assert s['meta']==meta
    missing,extra=net.load_state_dict(s['head'],strict=False)
    assert not extra and all(k.startswith('dino_encoder.') for k in missing)
    opt.load_state_dict(s['optimizer'])
    start=s['step']
    running=bytes.fromhex(s['sampling'])
    torch.cuda.set_rng_state(s['gpu_rng'])
    torch.set_rng_state(s['cpu_rng'])
    del s
for step in range(start,STEPS):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,step]))
    g=train[int(rng.integers(len(train)))]
    name=groups[g][int(rng.integers(len(groups[g])))][0]
    a,m=np.load(CACHE/g/(name+'.npy'))
    h,w=a.shape
    y=int(rng.integers(max(1,h-255)))
    x=int(rng.integers(max(1,w-255)))
    flip_y,flip_x=rng.integers(2,size=2)
    a=a[y:y+256,x:x+256].astype(np.float32)
    m=m[y:y+256,x:x+256].astype(np.int64)
    if flip_y: a,m=a[::-1],m[::-1]
    if flip_x: a,m=a[:,::-1],m[:,::-1]
    pad=((0,256-a.shape[0]),(0,256-a.shape[1]))
    image=torch.from_numpy(np.pad((a/255-.5)/.25,pad))[None,None].cuda()
    target=torch.from_numpy(np.pad(m,pad,constant_values=-1))[None].cuda()
    running=hashlib.sha256(running+repr((g,name,y,x,int(flip_y),int(flip_x))).encode()).digest()
    net.train()
    net.dino_encoder.eval()
    lr=1e-4*min(1,(step+1)/100)*(.1+.45*(1+np.cos(np.pi*step/STEPS)))
    for pg in opt.param_groups: pg['lr']=lr
    opt.zero_grad(set_to_none=True)
    loss=p.loss(p.forward(net,image),target)
    loss.backward()
    assert all(v.grad is None for v in net.dino_encoder.parameters())
    torch.nn.utils.clip_grad_norm_(params,12,error_if_nonfinite=True)
    opt.step()
    n=step+1
    if n==1 or n%100==0:
        atomic(out/'progress.json',dict(step=n,loss=loss.item(),sampling=running.hex()))
    if n in (10000,20000,40000):
        tmp=out/'checkpoint.tmp'
        torch.save(dict(meta=meta,step=n,head=p.head_state(net),optimizer=opt.state_dict(),sampling=running.hex(),cpu_rng=torch.get_rng_state(),gpu_rng=torch.cuda.get_rng_state()),tmp)
        tmp.replace(out/f'head_{n:05d}.pth')
net.eval()
def score(a,m):
    seg=p.sliding(net,a,tile_batch_size=16)
    counts=p.confusion(seg,m!=0)
    result=p.overlap(counts)
    denom=sum(counts[:3])
    result['iou']=counts[0]/denom if denom else 1.
    return result
cem=[]
for g in val:
    for name,_,_ in groups[g]:
        a,m=np.load(CACHE/g/(name+'.npy'))
        cem.append(dict(group=g,name=name,**score(a,m)))
atomic(out/'cem_validation.json',cem)
mb=[]
for r in json.loads((MB/'MB1_benchmark_manifest.json').read_text())['entries']:
    image=Planes(r['image_path'],r['shape_zyx'])
    label=Planes(r['label_path'],r['shape_zyx'])
    for z in range(r['shape_zyx'][0]):
        mb.append(dict(group=r['dataset'],name=r['id'],z=z,**score(image[z],label[z])))
    image.close()
    label.close()
assert len(mb)==2595
atomic(out/'mb1.json',mb)
atomic(out/'complete.json',dict(**meta,sampling=running.hex(),head_sha256=sha(out/'head_40000.pth'),cem_images=len(cem),mb_planes=len(mb)))
