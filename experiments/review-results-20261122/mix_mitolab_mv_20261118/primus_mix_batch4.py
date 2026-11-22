"""Matched frozen Primus readouts using the MitoNet mixed-source batch-four sampler."""
import os,sys,time
from pathlib import Path
import numpy as np
import torch
from batch4_prepare import OUT,CACHE4,STEPS,BATCH
from mixture import SEED,MB,read,sha,atomic
from fit import primus

ROOT=OUT/'primus_mix'
MODELS=('P0','B100','Barrier100')

def train(name,smoke=False):
    sys.path.insert(0,str(MB/'code'))
    from common import gpu_guard
    gpu=gpu_guard()
    out=ROOT/name
    out.mkdir(parents=True,exist_ok=True)
    ready=read(OUT/'ready.json')
    assert ready['batch']==BATCH==4 and ready['mixed_mv']==ready['mixed_cem']==STEPS*BATCH//2
    net,p,digest=primus(name)
    initial=p.state_hash(p.head_state(net))
    assert not any(v.requires_grad for v in net.dino_encoder.parameters())
    encoder_hash=p.state_hash(net.dino_encoder.state_dict())
    params=[v for v in net.parameters() if v.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.98),weight_decay=.05)
    raw=np.load(CACHE4/'mix/raw.npy',mmap_mode='r')
    labels=np.load(CACHE4/'mix/labels.npy',mmap_mode='r')
    assert raw.shape==labels.shape==(STEPS*BATCH,256,256)
    c=dict(model=name,seed=SEED,steps=STEPS,batch_size=BATCH,init_encoder_sha=digest,
           init_head_sha=initial,draws_sha=ready['draw_sha']['mix'],code_sha=sha(Path(__file__)))
    start=0
    states=sorted(out.glob('state_*.pt'))
    if states and not smoke:
        state=torch.load(states[-1],map_location='cpu',weights_only=True)
        assert state['meta']==c
        start=state['step']
        mi,ex=net.load_state_dict(torch.load(out/f'model_{start}.pt',map_location='cpu',weights_only=True),strict=False)
        assert not ex and all(k.startswith('dino_encoder.') for k in mi)
        opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['gpu_rng'])
        del state
    tick=time.monotonic()
    median=[]
    for step in range(start,12 if smoke else STEPS):
        sl=slice(step*BATCH,(step+1)*BATCH)
        a=raw[sl].astype(np.float32);ids=labels[sl]
        valid=ids>=0
        a[~valid]=127.5
        image=torch.from_numpy((a/255-.5)/.25)[:,None].cuda()
        target=torch.from_numpy(np.where(valid,ids>0,-1).astype(np.int64)).cuda()
        assert image.shape==(4,1,256,256)
        net.train();net.dino_encoder.eval()
        lr=1e-4*min(1,(step+1)/100)*(.1+.45*(1+np.cos(np.pi*step/STEPS)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        loss=p.loss(p.forward(net,image),target)
        loss.backward()
        assert all(v.grad is None for v in net.dino_encoder.parameters())
        torch.nn.utils.clip_grad_norm_(params,12,error_if_nonfinite=True)
        opt.step()
        median.append(float(loss.detach()))
        n=step+1
        if n==1 or n%100==0 or smoke:
            atomic(out/('smoke_progress.json' if smoke else 'progress.json'),dict(step=n,gpu=gpu,loss=float(np.median(median)),seconds=time.monotonic()-tick,peak_bytes=torch.cuda.max_memory_allocated()))
            median.clear()
        if n in (10000,20000,30000,40000):
            ck=out/f'model_{n}.pt';tmp=ck.with_suffix('.tmp');torch.save(p.head_state(net),tmp);tmp.replace(ck)
            ck=out/f'state_{n:05}.pt';tmp=ck.with_suffix('.tmp')
            torch.save(dict(meta=c,step=n,optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),gpu_rng=torch.cuda.get_rng_state()),tmp)
            tmp.replace(ck)
    assert encoder_hash==p.state_hash(net.dino_encoder.state_dict())
    if smoke:
        net.eval()
        a=raw[0];shape=p.sliding(net,a,tile_batch_size=16).shape
        assert shape==(256,256)
    else:
        atomic(out/'trained.json',dict(**c,checkpoint_sha=sha(out/'model_40000.pt'),encoder_unchanged=True))

if __name__=='__main__':
    mode,name=sys.argv[1:3]
    assert name in MODELS
    if mode=='evaluate':
        import evaluate
        evaluate.X=ROOT
        evaluate.run(name)
    else:train(name,mode=='smoke')
