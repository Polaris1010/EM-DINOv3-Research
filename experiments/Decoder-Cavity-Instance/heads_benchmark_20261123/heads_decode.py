import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F

from heads_model import CONFIG, PREVIOUS
sys.path.append(str(PREVIOUS/"followup_ablation"))
from bc_postprocess import boundary_watershed
from inference import CONNECT8, GAUSS, SIZE, starts


@torch.no_grad()
def predict(net, image, arm):
    h,w = image.shape
    coords = [(y,x) for y in starts(h,"stride120") for x in starts(w,"stride120")]
    out = np.zeros((4 if arm=="panoptic" else 3,h,w),np.float32)
    count = np.zeros((h,w),np.float32)
    for i in range(0,len(coords),16):
        tiles, regions = [], []
        for y,x in coords[i:i+16]:
            r = np.s_[y:y+SIZE,x:x+SIZE]
            a = np.asarray(image[r],np.float32)
            regions.append((r,a.shape))
            a = np.pad(a,((0,SIZE-a.shape[0]),(0,SIZE-a.shape[1])),mode="reflect")
            tiles.append((a/255-.5)/.25)
        z = net(torch.from_numpy(np.stack(tiles))[:,None].cuda()).float()
        if arm == "batch32":
            z = z.sigmoid()
        else:
            z[:,0].sigmoid_()
        for p, (r, (hh,ww)) in zip(z.cpu().numpy(),regions):
            weights = GAUSS[:hh,:ww]
            out[:,r[0],r[1]] += p[:,:hh,:ww]*weights
            count[r] += weights
    assert count.min()>0 and np.isfinite(out).all()
    return out/count


def min_area(ids, size):
    keep = np.bincount(ids.ravel()) >= size
    keep[0] = False
    return (np.cumsum(keep,dtype=np.int32)*keep)[ids]


def panoptic(maps, foreground, center_thresh):
    heat = maps[1]
    maxima = heat >= ndi.maximum_filter(heat,size=CONFIG["panoptic"]["nms"],mode="constant",cval=-np.inf)
    maxima &= heat > center_thresh
    cy,cx = np.nonzero(maxima)
    ids = np.zeros(heat.shape,np.int32)
    if not len(cy) or not foreground.any():
        return ids
    y,x = np.nonzero(foreground)
    votes = np.column_stack((y+maps[2,y,x],x+maps[3,y,x]))
    _, assigned = cKDTree(np.column_stack((cy,cx))).query(votes)
    ids[y,x] = assigned+1
    return ids


@torch.no_grad()
def flow_endpoints(flow, foreground, iterations):
    y,x = np.nonzero(foreground)
    if not len(y):
        return {}
    h,w = foreground.shape
    scale = torch.tensor([max(1,w-1),max(1,h-1)], device="cuda",dtype=torch.float32)
    coords = torch.from_numpy(np.stack((x,y),axis=-1)).to(device="cuda",dtype=torch.float32)
    points = (2*coords/scale-1)[None,None]
    field = torch.from_numpy(np.ascontiguousarray(flow[::-1])).cuda()[None]
    field = field * torch.from_numpy(foreground).cuda()
    field = field * (2*CONFIG["inference"]["cellpose_euler_step"]/scale)[None,:,None,None]
    result = {}
    select = set(iterations)
    for i in range(1,max(select)+1):
        delta = F.grid_sample(field,points,mode="bilinear",padding_mode="border",align_corners=True)
        points.add_(delta.permute(0,2,3,1)).clamp_(-1,1)
        if i in select:
            result[i] = ((points[0,0]+1)*scale/2).round().to(torch.int64).cpu().numpy()[:,::-1]
    return result


def endpoint_clusters(foreground, endpoints):
    y,x = np.nonzero(foreground)
    ids = np.zeros(foreground.shape,np.int32)
    if endpoints is None:
        return ids
    hist = np.zeros(foreground.shape,np.int32)
    np.add.at(hist,tuple(endpoints.T),1)
    cfg = CONFIG["cellpose"]
    peaks = (hist>=cfg["endpoint_histogram_min"]) & (hist==ndi.maximum_filter(hist,size=cfg["peak_window"]))
    plateaus,n = ndi.label(peaks,structure=CONNECT8)
    if not n:
        return ids
    seeds = ndi.maximum_position(hist,plateaus,np.arange(1,n+1))
    dist, assigned = cKDTree(np.asarray(seeds)).query(endpoints,p=np.inf,distance_upper_bound=cfg["endpoint_radius"]+0.1)
    ok = np.isfinite(dist)
    ids[y[ok],x[ok]] = assigned[ok]+1
    # A basin can contain disconnected foreground fragments. Give them distinct
    # identities so a single sink cannot join disconnected structures.
    from skimage.measure import label
    return label(ids,background=0,connectivity=2).astype(np.int32)


def decoded_grid(arm,maps,settings):
    by_fg = {}
    for threshold, control, size in settings:
        by_fg.setdefault(threshold,{}).setdefault(control,[]).append(size)
    for threshold, controls in by_fg.items():
        foreground = maps[0] > threshold
        ends = flow_endpoints(maps[1:],foreground,controls) if arm=="cellpose" else {}
        for control,sizes in controls.items():
            if arm=="batch32":
                for size in sizes:
                    yield (threshold,control,size), boundary_watershed(maps[0],maps[1],threshold,control,size)
                continue
            ids = (panoptic(maps,foreground,control) if arm=="panoptic"
                   else endpoint_clusters(foreground,ends.get(control)))
            for size in sizes:
                yield (threshold,control,size), min_area(ids,size)
