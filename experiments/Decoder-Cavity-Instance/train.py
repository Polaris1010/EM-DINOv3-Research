import json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import OUT,BASE,ARMS,build,semantic,SEED,sha,read,atomic,gpu_guard
from batch4_prepare import CACHE4,BATCH,STEPS
from targets import encode,verify_examples

def train(arm,smoke):
    gpu=gpu_guard()
    torch.set_num_threads(2)
    d=OUT/arm
    d.mkdir(exist_ok=True)
    encoder=OUT.parent/'pilot_c100_paired_20261028/runs/C1_Barrier_100k/checkpoints/step000100000/teacher_backbone.pth'
    model,p,encoder_sha=build(arm)
    assert sha(encoder)==encoder_sha and all(not w.requires_grad for w in model.enc.parameters())
    original=p.state_hash(model.enc.state_dict())
    init=p.state_hash({k:v for k,v in model.head.state_dict().items() if not k.startswith('aux.')})
    opt=torch.optim.AdamW(model.head.parameters(),lr=.0001,betas=(.9,.98),weight_decay=.05)
    raw,labels,valid_ids=[np.load(CACHE4/'cem'/n,mmap_mode='r') for n in ('raw.npy','labels.npy','instance.npy')]
    assert raw.shape==labels.shape==(STEPS*BATCH,256,256) and valid_ids.all()
    draw_sha=read(BASE/'batch4/ready.json')['draw_sha']['cem']
    assert sha(BASE/'batch4/cem_draws.jsonl')==draw_sha
    bcd=arm==ARMS[1]
    fixed=dict(arm=arm,seed=SEED,updates=STEPS,batch=BATCH,grad_accum=1,encoder_sha=encoder_sha,initial_shared_head_sha=init,
         draw_sha=draw_sha,weights={'foreground':1,'boundary':int(bcd),'distance':int(bcd),'distance_background':.05},
         source_sha={n:sha(OUT/n) for n in ('model.py','train.py','targets.py')})
    atomic(d/('smoke_config.json' if smoke else 'train_config.json'),fixed)
    t=time.monotonic()
    picks=verify_examples(raw,labels,d/'target_examples') if smoke and bcd else []
    start=0
    saved=sorted(d.glob('state_*.pt'))
    if saved and not smoke:
        state=torch.load(saved[-1],weights_only=True)
        assert state['config']==fixed
        opt.load_state_dict(state['optimizer'])
        model.head.load_state_dict(state['head'])
        start=state['step']
        del state
    torch.cuda.reset_peak_memory_stats()
    loss_values=[]
    limit=100 if smoke else STEPS
    for step in range(start,limit):
        sl=slice(step*BATCH,(step+1)*BATCH)
        ids=np.array(labels[sl]);v=ids>=0
        a=np.where(v,np.array(raw[sl],dtype=np.float32),127.5)
        image=torch.as_tensor((a/255-.5)/.25,device='cuda')[:,None]
        valid=torch.as_tensor(v,device='cuda')
        y=torch.as_tensor(ids>0,device='cuda')
        z=model(image)
        assert z.shape==(4,3 if bcd else 1,256,256) and not model.enc.training
        lf=semantic(z[:,0],y.float(),valid)
        terms={'fg':lf}
        loss=lf
        if bcd:
            b,dist=map(np.stack,zip(*(encode(x) for x in ids)))
            bd=torch.as_tensor(b,device='cuda',dtype=torch.float32)
            td=torch.as_tensor(dist,device='cuda')
            terms['bnd']=semantic(z[:,1],bd,valid)
            dloss=F.smooth_l1_loss(z[:,2].sigmoid(),td,reduction='none')
            fg=valid&y;bg=valid&~y
            terms['dist']=(dloss*fg).sum()/fg.sum().clamp_min(1)+.05*(dloss*bg).sum()/bg.sum().clamp_min(1)
            loss=loss+terms['bnd']+terms['dist']
        opt.param_groups[0]['lr']=1e-4*min(1,(step+1)/100)*(.1+.45*(1+np.cos(np.pi*step/STEPS)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert all(w.grad is None for w in model.enc.parameters())
        grad=torch.nn.utils.clip_grad_norm_(model.head.parameters(),12,error_if_nonfinite=True)
        assert torch.isfinite(loss) and grad>0
        opt.step()
        n=step+1
        loss_values.append(float(loss.detach()))
        if n==1 or n%100==0 or smoke and n%20==0:
            record=dict(step=n,gpu=gpu,loss=float(np.median(loss_values[-100:])),seconds=time.monotonic()-t,
                  peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),**{k:float(v.detach()) for k,v in terms.items()})
            print(json.dumps(record),flush=True)
            atomic(d/('smoke_progress.json' if smoke else 'progress.json'),record)
        if not smoke and n%5000==0:
            ck=d/f'state_{n:05d}.pt'
            tmp=ck.with_suffix('.tmp')
            torch.save(dict(config=fixed,step=n,head=model.head.state_dict(),optimizer=opt.state_dict()),tmp)
            tmp.replace(ck)
    assert original==p.state_hash(model.enc.state_dict()) and sha(encoder)==encoder_sha
    free,total=torch.cuda.mem_get_info()
    peak=torch.cuda.max_memory_reserved()
    margin=free+torch.cuda.memory_reserved()-peak
    final=dict(**record,encoder_unchanged=True,headroom_bytes=margin,init_shared_head_sha=init)
    if smoke:
        early=float(np.median(loss_values[:20]));late=float(np.median(loss_values[-20:]))
        assert margin>=1.5*1024**3 and late<early,(margin,early,late)
        atomic(d/'smoke_pass.json',dict(**final,first20_median=early,last20_median=late,target_examples=picks))
    else:
        out=d/'model_40000.pt'
        tmp=out.with_suffix('.tmp')
        torch.save(model.head.state_dict(),tmp)
        tmp.replace(out)
        atomic(d/'trained.json',dict(**final,model_sha256=sha(out),config=fixed))
if __name__=='__main__':
    mode,arm=sys.argv[1:]
    assert mode in ('smoke','train') and arm in ARMS
    train(arm,mode=='smoke')
