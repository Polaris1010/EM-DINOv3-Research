import csv
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_erosion,binary_fill_holes
OUT=Path(__file__).resolve().parent
BASE=OUT.parent.parent/"mix_mitolab_mv_20261118"
sys.path.insert(0,str(BASE))
from mixture import MB,read,sha,atomic
from fit import primus
from c0_matched import encoder
sys.path.insert(0,str(MB/"code"))
from common import Planes,gpu_guard

def select():
    volumes=defaultdict(list)
    for row in read(MB/"MB1_benchmark_manifest.json")["entries"]:
        volumes[row["dataset"]].append(row)
    for group,rows in volumes.items():
        if group=="TEM-100":
            rows=sorted(rows,key=lambda r:r["id"])
            for i in np.linspace(0,len(rows)-1,10,dtype=int):
                yield rows[i],0
        else:
            assert len(rows)==1
            r=rows[0]
            for z in np.linspace(0,r["shape_zyx"][0]-1,7,dtype=int):
                yield r,int(z)

def square(a,size,value):
    # Align each size to the identical native-image center using integer origins.
    h,w=a.shape
    y,x=(h-size)//2,(w-size)//2
    crop=a[max(0,y):min(h,y+size),max(0,x):min(w,x+size)]
    return np.pad(crop,((max(0,-y),max(0,y+size-h)),(max(0,-x),max(0,x+size-w))),constant_values=value)

@torch.inference_mode()
def run(name):
    gpu=gpu_guard()
    torch.set_num_threads(2)
    if name=="C0_Vanilla_100k":
        net,p,digest=encoder(name)
        root=BASE/"batch4/c0_matched/cem"/name
    else:
        net,p,digest=primus(name)
        root=BASE/"batch4/primus_cem"/name
    ck=root/"model_40000.pt"
    meta=read(root/"trained.json")
    assert digest==meta["init_encoder_sha"] and sha(ck)==meta["checkpoint_sha"] and meta["seed"]==20261014
    missing,extra=net.load_state_dict(torch.load(ck,map_location="cpu",weights_only=True),strict=False)
    assert not extra and all(x.startswith("dino_encoder.") for x in missing)
    original_name=name
    folder=OUT.parent/"runs"/name/"Gram/checkpoints/step000120000"
    checkpoint=folder/"teacher_backbone.pth"
    receipt=read(folder/"teacher_backbone.json")
    assert receipt["step"]==120000 and sha(checkpoint)==receipt["sha256"]
    net.dino_encoder.load_state_dict(torch.load(checkpoint,map_location="cpu",mmap=True,weights_only=True)["backbone"],strict=True)
    name=original_name+"_Gram120k"
    net.eval()
    rows=[]
    for r,z in select():
        img,gt=[Planes(r[k],r["shape_zyx"]) for k in ("image_path","label_path")]
        a,b=img[z],gt[z]
        valid=square(np.ones(a.shape,bool),256,False)
        truth=square(b>0,256,False)
        # Erode full GT first, avoiding an artificial boundary at the score crop.
        inner=square(binary_erosion(b>0,iterations=3),256,False)
        for size in (256,512,1024):
            pixels=square(a.astype(np.float32),size,127.5)
            x=torch.from_numpy((pixels/255-.5)/.25)[None,None].cuda()
            logit=p.forward(net,x)
            assert logit.shape==(1,2,size,size) and torch.isfinite(logit).all()
            offset=(size-256)//2
            box=np.s_[offset:offset+256,offset:offset+256]
            margin=(logit[0,1]-logit[0,0]).cpu().numpy()[box]
            pred=(margin>0)&valid
            holes=binary_fill_holes(pred)&~pred
            rows.append(dict(group=r["dataset"],id=r["id"],z=z,window=size,
                             padding=min(a.shape)<size,n=int(valid.sum()),gt=int(truth.sum()),
                             tp=int((pred&truth).sum()),fp=int((pred&~truth).sum()),
                             inner=int(inner.sum()),inner_missed=int((inner&~pred).sum()),
                             enclosed_missed=int((inner&holes).sum()),fg_margin_sum=float(margin[truth].sum())))
        img.close()
        gt.close()
    atomic(OUT/(name+".json"),dict(model=name,gpu=gpu,head_sha=meta["checkpoint_sha"],encoder_sha=receipt["sha256"],rows=rows))

if __name__=="__main__":
    for name in sys.argv[1:]:
        run(name)
