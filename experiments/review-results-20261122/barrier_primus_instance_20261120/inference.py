import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from model import capture

@torch.inference_mode()
def predict(net, probe, stage, head, image):
    # Run the unchanged sliding-window semantic implementation. Observe the same tiles
    # through a read-only hook. Fuse fields on the native image before detecting centers.
    h,w=image.shape
    coords=[(y,x) for y in probe.starts(h) for x in probe.starts(w)]
    total=np.zeros((3,h,w),np.float32)
    weight=np.zeros((h,w),np.float32)
    taper=np.maximum(.02,np.hanning(256)).astype(np.float32)
    window=taper[:,None]*taper[None,:]
    pos=0
    def collect(module,inputs,feature):
        nonlocal pos
        maps=head(feature).float().cpu().numpy()
        for z in maps:
            y,x=coords[pos]
            pos+=1
            region=np.s_[y:min(y+256,h),x:min(x+256,w)]
            hh,ww=weight[region].shape
            win=window[:hh,:ww]
            total[:,region[0],region[1]]+=z[:,:hh,:ww]*win
            weight[region]+=win
    handle=stage.register_forward_hook(collect)
    try:
        sem=probe.sliding(net,image,tile_batch_size=16)
    finally:
        handle.remove()
    assert pos==len(coords) and weight.min()>0
    total/=weight
    peaks=(total[0]>=.1)&(total[0]==ndi.maximum_filter(total[0],size=9))
    islands,n=ndi.label(peaks)
    ids=np.zeros((h,w),np.int32)
    yy,xx=np.nonzero(sem)
    if len(yy) and n:
        centers=ndi.maximum_position(total[0],islands,np.arange(1,n+1))
        tree=cKDTree(centers)
        for st in range(0,len(yy),262144):
            y,x=yy[st:st+262144],xx[st:st+262144]
            projected=np.stack((y+32*total[1,y,x],x+32*total[2,y,x]),axis=1)
            ids[y,x]=tree.query(projected)[1]+1
    elif len(yy):
        ids=ndi.label(sem,structure=np.ones((3,3),int))[0].astype(np.int32)
    assert np.array_equal(ids>0,sem)
    return sem,ids,bool(len(yy) and not n)
