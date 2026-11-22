from pathlib import Path
from itertools import product
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from model import OUT,ARMS,build,sha,read
SIZE=256
x=np.arange(SIZE,dtype=np.float32)-(SIZE-1)/2
GAUSS=np.exp(-(x[:,None]**2+x[None,:]**2)/(2*64**2)).astype(np.float32)
CONNECT8=np.ones((3,3),bool)
SETTINGS=('stride120','baseline128')
GRID=list(product((.25,.4),(.5,.65),(16,32)))

def starts(n,mode):
    if mode=='baseline128':
        return sorted(set(range(0,max(1,n-255),128))|{max(0,n-256)})
    return list(range(0,120*max(0,(n-256+119)//120)+1,120))

@torch.no_grad()
def predict(net,image,mode):
    h,w=image.shape
    coords=[(y,x) for y in starts(h,mode) for x in starts(w,mode)]
    c=3 if net.head.aux is not None else 1
    out=np.zeros((c,h,w),np.float32)
    counts=np.zeros((h,w),np.float32)
    for i in range(0,len(coords),16):
        tiles=[];regions=[]
        for y,x in coords[i:i+16]:
            r=np.s_[y:y+256,x:x+256]
            a=np.asarray(image[r],np.float32)
            regions.append((r,a.shape))
            a=np.pad(a,((0,256-a.shape[0]),(0,256-a.shape[1])),mode='reflect')
            tiles.append((a/255-.5)/.25)
        probs=net(torch.from_numpy(np.stack(tiles))[:,None].cuda()).sigmoid().cpu().numpy()
        assert np.isfinite(probs).all()
        for p,(r,(hh,ww)) in zip(probs,regions):
            weights=GAUSS[:hh,:ww] if mode=='stride120' else 1.0
            out[:,r[0],r[1]]+=p[:,:hh,:ww]*weights
            counts[r]+=weights
    assert counts.min()>0
    return out/counts

def cc(fg):
    return ndi.label(fg,structure=CONNECT8)[0]

def native(maps,params,valid=None):
    seed,bnd,area=params
    fg=maps[0]>.5
    if valid is not None:fg &= valid
    core=fg&(maps[1]<bnd)
    dist=ndi.gaussian_filter(maps[2],sigma=1.0)
    co=peak_local_max(dist,threshold_abs=seed,min_distance=3,exclude_border=False,labels=core.astype(np.uint8))
    if not len(co):return np.zeros(core.shape,np.int32)
    seeds=np.zeros(core.shape,np.int32)
    seeds[tuple(co.T)]=np.arange(1,len(co)+1)
    ids=watershed(-dist,seeds,mask=core,connectivity=CONNECT8)
    ok=np.bincount(ids.ravel())>=area
    ok[0]=False
    relabel=np.cumsum(ok)*ok
    return relabel[ids].astype(np.int32)

def load(arm):
    net,probe,enc=build(arm)
    d=OUT/arm
    rec=read(d/'trained.json')
    ck=d/'model_40000.pt'
    assert rec['step']==40000 and sha(ck)==rec['model_sha256'] and enc==rec['config']['encoder_sha']
    net.head.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=True)
    net.eval()
    return net,rec
