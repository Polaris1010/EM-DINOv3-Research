from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, os
import numpy as np
import tifffile
X=Path(__file__).resolve().parent
ROOT=X.parent.parent
D=ROOT/'MitoNet/dataset/EMPIAR-11037/cem_mitolab'
CACHE=Path('/opt/cem_mitolab_20261118')
def digest(name):
    return hashlib.sha256(('20261014:'+name).encode()).hexdigest()
def prepare(group):
    imgs={p.name:p for p in (group/'images').glob('*.tiff')}
    masks={p.name:p for p in (group/'masks').glob('*.tiff')}
    assert imgs and imgs.keys()==masks.keys(),group
    out=CACHE/group.name
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in sorted(imgs):
        a=tifffile.imread(imgs[name])
        m=tifffile.imread(masks[name])
        if a.ndim!=2 or a.dtype!=np.uint8 or m.shape!=a.shape or m.dtype.kind not in 'uib':
            raise ValueError((imgs[name],a.shape,a.dtype,m.shape,m.dtype))
        fg=m!=0
        np.save(out/(name+'.npy'),np.stack([a,fg.astype(np.uint8)]),allow_pickle=False)
        rows.append([name,list(a.shape),int(fg.sum())])
    return group.name,rows
if __name__=='__main__':
    groups=sorted(g for g in D.iterdir() if g.is_dir() and (g/'images').is_dir())
    assert len(groups)>1
    val=sorted(g.name for g in sorted(groups,key=lambda g:digest(g.name))[:max(1,round(.05*len(groups)))])
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows=dict(pool.map(prepare,groups))
    index={'groups':rows,'validation':val,'seed':20261014,'validation_rule':'first round(5% of collections) by SHA256 of seed:collection; all images including empty masks','target':'stored instance ID != 0'}
    tmp=X/'manifest.tmp'
    tmp.write_text(json.dumps(index,sort_keys=True))
    tmp.replace(X/'manifest.json')
    print(len(groups),len(val),sum(map(len,rows.values())),sum(len(rows[k]) for k in val),flush=True)
