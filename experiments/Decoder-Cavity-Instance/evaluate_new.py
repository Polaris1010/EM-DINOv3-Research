import hashlib,json,statistics as st,sys
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from model import OUT,ARMS,read,sha,atomic,gpu_guard
from inference import load,predict,cc,native,GRID,SETTINGS
from bench_inputs import cem_items,cem_read,datasets
from instance_repeat import score
sys.path.append(str(OUT.parent/'barrier_primus_instance_20261120'))
from evaluate_head import touching
sys.path.insert(0,str(OUT.parent/'square_hole_diagnosis_20261122'/'gt_interior_holes'))
from core import gt_maps,metrics as interior
from evaluate import metrics as semantic
sys.path.insert(0,str(OUT.parent/'square_hole_diagnosis_20261122'/'fusion_trace'))
from trace import shapes
from probe2 import holes_of

def choose(net,rec):
    d=OUT/ARMS[1]
    f=d/'watershed_locked.json'
    sums=defaultdict(lambda:defaultdict(list))
    used=[]
    if f.exists():
        old=read(f)
        assert old['checkpoint_sha']==rec['model_sha256'] and old['source_sha']==sha(OUT/'inference.py')
        return old['parameters']
    # Fixed ordering on image IDs only; exactly six (or fewer) per held-out collection.
    groups=list(cem_items())
    for g,keys in groups:
        selected=sorted(keys,key=lambda x:hashlib.sha256(('20261014/'+x[0]).encode()).hexdigest())[:6]
        for name,_ in selected:
            a,gt=cem_read(g,name)
            maps=predict(net,a,'stride120')
            for setting in GRID:
                m=score(gt,native(maps,setting))
                sums[setting][g].append((m['PQ'],m['AP_75']))
            used.append([g,name])
        print('tune',g,len(used),flush=True)
    ranked=[]
    for p,v in sums.items():
        ranked.append(dict(parameters=p,PQ=st.mean(st.mean(x[0] for x in a) for a in v.values()),
                                AP75=st.mean(st.mean(x[1] for x in a) for a in v.values())))
    ranked.sort(key=lambda v:(-v['PQ'],-v['AP75'],v['parameters']))
    atomic(f,dict(checkpoint_sha=rec['model_sha256'],source_sha=sha(OUT/'inference.py'),data='cem_validation',rule='maximum collection-equal PQ, tie AP75, then lexicographic grid',examples=used,grid=ranked,parameters=ranked[0]['parameters']))
    return ranked[0]['parameters']

def stats(gt,ids):
    x=score(gt,ids)
    together,separated=touching(gt,ids)
    return dict(AP50=float(x['AP_50']),AP75=float(x['AP_75']),PQ=float(x['PQ']),
                MAE=int(abs(x['n_gt']-x['n_pred'])),touching_pairs=together,separated_pairs=separated)

def hq_row(mask):
    bl=holes_of(mask)
    n=int(bl.sum())
    return dict(fg=int(mask.sum()),enclosed_px=n,hole_ratio=n/max(1,int(mask.sum())+n),square_holes=len(shapes(mask)[0]))

def confusion(mask,gt,valid):
    m=mask if valid is None else mask[valid]
    t=gt>0 if valid is None else gt[valid]>0
    tp,ps,gs=int((m&t).sum()),int(m.sum()),int(t.sum())
    return [tp,ps-tp,gs-tp,m.size-ps-gs+tp]

@torch.no_grad()
def main(arm):
    gpu=gpu_guard()
    torch.set_num_threads(2)
    net,rec=load(arm)
    params=choose(net,rec) if arm==ARMS[1] else None
    d=OUT/arm/'evaluations'
    start=__import__('time').monotonic()
    for mode in SETTINGS:
        for split,group,key,planes in datasets():
            f=d/mode/split/(key+'.jsonl')
            part=f.with_suffix('.partial')
            if f.exists():continue
            f.parent.mkdir(parents=True,exist_ok=True)
            done=len(part.read_text().splitlines()) if part.exists() else 0
            with part.open('a') as handle:
                for i,(name,z,image,gt,valid,reliable) in enumerate(planes):
                    if i<done:continue
                    maps=predict(net,image,mode)
                    fg=maps[0]>.5
                    r=dict(arm=arm,split=split,group=group,id=name,z=z,window=mode)
                    if split=='hq':
                        r.update(hq_row(fg))
                        target=OUT/'hq_png'/arm/mode/f'{name}.png'
                        target.parent.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(fg.astype(np.uint8)*255).save(target)
                    else:
                        r['semantic']=semantic(confusion(fg,gt,valid))
                        if split=='mb1':
                            r['cavity']=interior(gt_maps(gt),fg)[0]
                        if reliable:
                            if valid is not None:gt=np.where(valid,gt,0)
                            r['cc']=stats(gt,cc(fg if valid is None else fg&valid))
                            if params is not None:r['watershed']=stats(gt,native(maps,params,valid))
                    handle.write(json.dumps(r)+'\n')
                    handle.flush()
                    if i%50==0:print(arm,mode,split,key,i+1,flush=True)
            part.replace(f)
    atomic(OUT/arm/'evaluation_pass.json',dict(gpu=gpu,seconds=__import__('time').monotonic()-start,
         checkpoint=rec['model_sha256'],settings=list(SETTINGS),watershed=params))

if __name__=='__main__':
    main(sys.argv[1])
