import json
import time
from model import BASE, OUT, SEED, build, capture, targets, torch, np, sha, read
from mixture import atomic
from batch4_prepare import CACHE4

def main():
    net,probe,stage,head,receipt = build()
    crops,labels,modes = [np.load(CACHE4/"cem"/n,mmap_mode="r") for n in ("raw.npy","labels.npy","instance.npy")]
    assert crops.shape == labels.shape == (160000,256,256) and modes.all()
    opt = torch.optim.AdamW(head.parameters(),lr=.0001,weight_decay=.05)
    steps = 40000
    config = dict(seed=SEED,batch=4,steps=steps,baseline=receipt,draw_sha=read(BASE/"batch4/ready.json")["draw_sha"]["cem"],
                  input="all 160000 CEM training draws",sigma_px=6,offset_unit_px=32,center_weight="1+20*target",
                  adamw_lr=.0001,weight_decay=.05,source_sha={n:sha(OUT/n) for n in ("model.py","train.py")})
    start = 0
    ck = OUT/"resume.pt"
    if ck.exists():
        state = torch.load(ck,weights_only=True)
        previous = state["config"]
        if previous["steps"] == 10000:
            assert state["step"] == 10000
            assert previous["source_sha"]["train.py"] == "f21cda3761c3d99f627fa51910c2e4e8c83fe5b0afa0935d3a6965c95299fcbc"
            assert previous["source_sha"]["model.py"] == config["source_sha"]["model.py"]
            assert all(previous[k] == config[k] for k in config if k not in ("steps","input","source_sha"))
            assert sha(OUT/"instance_10000.pt") == read(OUT/"trained.json")["sha256"]
        else:
            assert previous == config
        head.load_state_dict(state["head"])
        opt.load_state_dict(state["opt"])
        start = state["step"]
    atomic(OUT/"config_40000.json",config)
    begin = time.monotonic()
    for i in range(start,steps):
        sl = slice(4*i,4*i+4)
        ids = torch.from_numpy(np.array(labels[sl])).cuda()
        raw = torch.from_numpy(np.array(crops[sl],dtype=np.float32)).cuda()
        x = ((torch.where(ids>=0,raw,127.5)/255-.5)/.25)[:,None]
        if i == start:
            with torch.no_grad():
                original = probe.forward(net,x)
            features, hook = capture(stage)
        with torch.no_grad():
            logits = probe.forward(net,x)
        pred = head(features.pop())
        ht,of,val,fg = map(torch.stack,zip(*(targets(ids_j) for ids_j in ids)))
        err = (pred[:,0]-ht).square()*(1+20*ht)
        lc = (err*val).sum()/val.sum().clamp_min(1)
        lo = ((pred[:,1:]-of).abs()*fg[:,None]).sum()/(2*fg.sum()).clamp_min(1)
        opt.zero_grad(set_to_none=True)
        (lc+lo).backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(),12,error_if_nonfinite=True)
        opt.step()
        n = i+1
        if i == start:
            with torch.no_grad():
                assert torch.equal(original,logits) and torch.equal(original,probe.forward(net,x))
            features.clear()
            assert all(v.grad is None for v in net.parameters())
            atomic(OUT/"semantic_smoke.json",dict(step=n,equal_logits=True,pixels=int(ids.numel()),instance_pixels=int(fg.sum())))
        if n==1 or n%50==0:
            row=dict(step=n,center=float(lc),offset=float(lo),grad_norm=float(norm),seconds=time.monotonic()-begin)
            print(json.dumps(row),flush=True)
            atomic(OUT/"progress.json",row)
        if n%500==0:
            tmp=OUT/"resume.tmp"
            torch.save(dict(config=config,step=n,head=head.state_dict(),opt=opt.state_dict()),tmp)
            tmp.replace(ck)
    torch.save(head.state_dict(),OUT/"instance_40000.pt")
    atomic(OUT/"trained_40000.json",dict(steps=steps,sha256=sha(OUT/"instance_40000.pt")))

if __name__=="__main__":
    main()
