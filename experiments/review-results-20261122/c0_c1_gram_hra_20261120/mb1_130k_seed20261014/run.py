from collections import defaultdict
from pathlib import Path
import csv
import sys
import time
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
BASE=OUT.parent.parent/"mix_mitolab_mv_20261118"
sys.path.insert(0,str(BASE))
from mixture import MB, SEED, atomic, read, sha
import primus_cem_batch4 as protocol
from fit import primus
from evaluate import metrics
sys.path.insert(0,str(MB/"code"))
from common import Planes, gpu_guard
MODELS=("C0_Vanilla_100k", "C1_Barrier_100k")

def build(name):
    checkpoint=OUT.parent/"runs"/name/"HRA/checkpoints/step000130000/teacher_backbone.pth"
    info=read(checkpoint.with_suffix(".json"))
    digest=sha(checkpoint)
    assert digest==info["sha256"] and info["step"]==130000
    net,probe,_=primus("P0")
    weights=torch.load(checkpoint,map_location="cpu",mmap=True,weights_only=True)
    net.dino_encoder.load_state_dict(weights["backbone"],strict=True)
    return net,probe,digest

def slide512(net,probe,img):
    size,stride=512,256
    h,w=img.shape
    def starts(n):
        return sorted(set(range(0,max(1,n-size+1),stride))|{max(0,n-size)})
    coords=[(y,x) for y in starts(h) for x in starts(w)]
    acc=np.zeros((h,w),np.float32)
    weight=np.zeros((h,w),np.uint16)
    for idx in range(0,len(coords),4):
        tiles=[]
        for y,x in coords[idx:idx+4]:
            a=img[y:y+size,x:x+size].astype(np.float32)
            tiles.append((np.pad(a,((0,size-a.shape[0]),(0,size-a.shape[1])),constant_values=127.5)/255-.5)/.25)
        z=torch.from_numpy(np.stack(tiles))[:,None].cuda()
        logits=probe.forward(net,z)
        assert logits.shape==(len(tiles),2,size,size)
        probs=logits.softmax(1)[:,1].cpu().numpy()
        assert np.isfinite(probs).all()
        for p,(y,x) in zip(probs,coords[idx:idx+4]):
            hh,ww=min(h-y,size),min(w-x,size)
            acc[y:y+hh,x:x+ww]+=p[:hh,:ww]
            weight[y:y+hh,x:x+ww]+=1
    assert weight.min()>0
    return acc/weight>.5

@torch.inference_mode()
def evaluate(name):
    card=gpu_guard()
    torch.set_num_threads(2)
    d=OUT/name
    receipt=read(d/"trained.json")
    ck=d/"model_40000.pt"
    assert receipt["seed"]==SEED==20261014 and receipt["steps"]==40000 and sha(ck)==receipt["checkpoint_sha"]
    net,p,digest=build(name)
    assert digest==receipt["init_encoder_sha"]
    missing,extra=net.load_state_dict(torch.load(ck,map_location="cpu",weights_only=True),strict=False)
    assert not extra and all(k.startswith("dino_encoder.") for k in missing)
    net.eval()
    entries=read(MB/"MB1_benchmark_manifest.json")["entries"]
    for size in (256,512):
        pred=(lambda a:p.sliding(net,a,tile_batch_size=16)) if size==256 else (lambda a:slide512(net,p,a))
        directory=d/f"window_{size}"
        directory.mkdir(exist_ok=True)
        for r in entries:
            dest=directory/(r["id"]+".json")
            if dest.exists():
                continue
            a,b=[Planes(r[key],r["shape_zyx"]) for key in ("image_path","label_path")]
            scores=[]
            for z in range(r["shape_zyx"][0]):
                s=torch.as_tensor(pred(a[z]),device="cuda")
                t=torch.as_tensor(np.asarray(b[z])>0,device="cuda")
                assert s.shape==t.shape
                tp,fg,ps=torch.stack(((s&t).sum(),t.sum(),s.sum())).tolist()
                scores.append(dict(z=z,**metrics([tp,ps-tp,fg-tp,t.numel()-ps-fg+tp])))
                if z%25==0:
                    atomic(d/"eval_progress.json",dict(window=size,group=r["dataset"],id=r["id"],plane=z+1,planes=r["shape_zyx"][0],gpu=card))
            a.close()
            b.close()
            atomic(dest,dict(id=r["id"],group=r["dataset"],window=size,rows=scores))
        print(f"{name} window {size} complete",flush=True)
    summarize(name)

def summarize(name):
    grouped=defaultdict(list)
    fields=("dice","iou","precision","recall")
    for w in (256,512):
        for f in (OUT/name/f"window_{w}").glob("*.json"):
            d=read(f)
            grouped[w,d["group"]].extend(d["rows"])
        assert sum(len(r) for (s,_),r in grouped.items() if s==w)==2595
    result=[]
    for (w,g),rows in sorted(grouped.items()):
        result.append(dict(window=w,group=g,planes=len(rows),**{k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in fields}))
    for w in (256,512):
        group=[r for r in result if r["window"]==w and r["group"]!="ALL"]
        assert len(group)==7
        result.append(dict(window=w,group="ALL",planes=2595,**{k:float(np.mean([r[k] for r in group])) for k in fields}))
    tmp=OUT/name/"summary.tmp"
    with tmp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    tmp.replace(OUT/name/"summary.csv")

if __name__=="__main__":
    mode,name=sys.argv[1:]
    assert name in MODELS
    if mode=="train":
        protocol.ROOT=OUT
        protocol.primus=build
        if not (OUT/name/"trained.json").exists():
            protocol.train(name)
    else:
        evaluate(name)
