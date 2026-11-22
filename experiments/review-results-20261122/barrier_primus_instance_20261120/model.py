from pathlib import Path
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
BASE = Path(os.environ.get("SEMANTIC_RUN", OUT.parent / "mix_mitolab_mv_20261118"))
sys.path.insert(0, str(BASE))
from fit import primus
from mixture import SEED, read, sha, MB
sys.path.insert(0, str(MB / "code"))
from common import gpu_guard

class InstanceHead(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(channels, 64, 1), nn.GroupNorm(8, 64), nn.GELU(),
                         nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.GroupNorm(8, 64), nn.GELU(),
                         nn.Conv2d(64, 3, 1))

    def forward(self, feature):
        return F.interpolate(super().forward(feature), scale_factor=4, mode="bilinear", align_corners=False)

def build():
    gpu_guard()
    torch.set_num_threads(2)
    net, probe, digest = primus("Barrier100")
    folder = BASE / "batch4/primus_cem/Barrier100"
    receipt = read(folder / "trained.json")
    ck = folder / "model_40000.pt"
    assert digest == receipt["init_encoder_sha"] and sha(ck) == receipt["checkpoint_sha"]
    missing, extra = net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=False)
    assert not extra and all(k.startswith("dino_encoder.") for k in missing)
    net.eval().requires_grad_(False)
    stage = net.up_projection.decode[1]
    torch.manual_seed(SEED)
    return net, probe, stage, InstanceHead(stage[0].out_channels).cuda(), receipt

def capture(stage):
    features = []
    def hook(module, inputs, output):
        features[:] = [output.detach()]
    return features, stage.register_forward_hook(hook)

def targets(ids):
    # A padded pixel or crop boundary can hide the true centroid. Exclude entire affected IDs.
    valid = ids >= 0
    invalid_neighbors = F.max_pool2d((~valid).float()[None,None],3,1,1)[0,0].bool()
    edge = invalid_neighbors.clone()
    edge[[0,-1],:] = True
    edge[:,[0,-1]] = True
    truncated = ids[edge & (ids > 0)].unique()
    valid = valid & ~torch.isin(ids,truncated)
    fg = valid & (ids > 0)
    h,w = ids.shape
    yy,xx = torch.meshgrid(torch.arange(h,device=ids.device),torch.arange(w,device=ids.device),indexing="ij")
    coords = torch.stack((yy,xx)).float()
    heat = torch.zeros_like(yy,dtype=torch.float32)
    offsets = torch.zeros_like(coords)
    if fg.any():
        _, inv, count = torch.unique(ids[fg],return_inverse=True,return_counts=True)
        xy = coords[:,fg]
        centers = xy.new_zeros(2,len(count)).scatter_add_(1,inv.expand(2,-1),xy)/count
        offsets[:,fg] = (centers[:,inv]-xy)/32
        delta = torch.arange(-18,19,device=ids.device)
        dy,dx = torch.meshgrid(delta,delta,indexing="ij")
        cy,cx = centers.round().long()
        y,x = cy[:,None]+dy.flatten(),cx[:,None]+dx.flatten()
        keep = (y>=0)&(x>=0)&(y<h)&(x<w)
        values = torch.exp(-(dy.square()+dx.square()).float()/72).flatten().expand_as(y)
        heat.view(-1).scatter_reduce_(0,(y*w+x)[keep],values[keep],reduce="amax")
    return heat,offsets,valid,fg
