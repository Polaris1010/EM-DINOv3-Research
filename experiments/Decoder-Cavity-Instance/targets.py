import numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw

FOOTPRINT=ndi.generate_binary_structure(2,1)

def encode(ids):
    fg=ids>0
    bnd=np.zeros(ids.shape,bool)
    dist=np.zeros(ids.shape,np.float32)
    _,inv=np.unique(ids[fg],return_inverse=True)
    dense=np.zeros(ids.shape,np.int32)
    dense[fg]=inv+1
    for k,s in enumerate(ndi.find_objects(dense),1):
        obj=dense[s]==k
        ring=obj&~ndi.binary_erosion(obj,structure=FOOTPRINT,iterations=2,border_value=0)
        bnd[s][ring]=True
        edt=ndi.distance_transform_edt(np.pad(obj,1))[1:-1,1:-1]
        dist[s][obj]=(edt[obj]/max(1,edt.max())).astype(np.float32)
    assert bnd[~fg].sum()==0 and dist[~fg].sum()==0 and (dist[fg]>0).all() and dist.max()<=1
    return bnd,dist

def contact(ids):
    return any(np.any((a>0)&(b>0)&(a!=b)) for a,b in ((ids[:-1],ids[1:]),(ids[:,:-1],ids[:,1:])))

def verify_examples(raw,labels,out):
    out.mkdir(exist_ok=True)
    picks=[]
    for i in range(len(labels)):
        ids=labels[i]
        if not contact(ids):continue
        b,d=encode(ids)
        valid=(ids>=0)
        for a,k in ((ids[:-1],ids[1:]),(ids[:,:-1],ids[:,1:])):
            line=(a>0)&(k>0)&(a!=k)
            observed=b[:-1] if a.shape[0]<ids.shape[0] else b[:,:-1]
            assert np.all(observed[line])
        cmap=np.random.default_rng(123).integers(25,255,(4096,3),dtype=np.uint8)
        colored=np.where((ids>0)[...,None],cmap[ids%4096],0)
        img=Image.new('RGB',(1024,280),'#222222')
        draw=ImageDraw.Draw(img)
        for j,(name,p) in enumerate(zip(('Raw','Instance IDs','2 px boundary','Distance'),(raw[i],colored,b.astype(np.uint8)*255,(255*d).astype(np.uint8)))):
            img.paste(Image.fromarray(p).convert('RGB'),(j*256,24))
            draw.text((j*256+4,4),name,fill='white')
        img.save(out/f'draw_{i}.png')
        picks.append(i)
        if len(picks)==5:break
    assert len(picks)==5
    return picks
