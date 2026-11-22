from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'mix_mitolab_mv_20261118'
sys.path.insert(0,str(BASE))
from fit import primus
from mixture import MB,SEED,sha,read,atomic
sys.path.insert(0,str(MB/'code'))
from common import gpu_guard

class Conv(nn.Sequential):
    def __init__(self,c,d,k=3):
        super().__init__(nn.Conv2d(c,d,k,padding=k//2,bias=False),nn.GroupNorm(8,d),nn.GELU())

class Spatial(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv=nn.Sequential(Conv(c,c),nn.Conv2d(c,c,3,padding=1,bias=False),nn.GroupNorm(8,c))
    def forward(self,x):return F.gelu(x+self.conv(x))

class MultiScaleContextDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.ModuleList(Conv(1024,256,1) for _ in range(4))
        self.context=nn.Sequential(Spatial(256),Spatial(256))
        self.raw=nn.Sequential(Conv(1,16),Conv(16,32))
        self.up=nn.ModuleList([Conv(256,128),Conv(128,64),Conv(64+32,32),Conv(32+32,32)])
    def forward(self,features,image):
        x=self.context(torch.stack([p(f.float()) for p,f in zip(self.project,features)]).mean(0))
        raw=self.raw(image)
        for i,layer in enumerate(self.up):
            x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
            if i>=2:
                skip=F.avg_pool2d(raw,2) if i==2 else raw
                x=torch.cat([x,skip],1)
            x=layer(x)
        return x

class Head(nn.Module):
    def __init__(self,bcd):
        super().__init__()
        self.decoder=MultiScaleContextDecoder()
        self.fg=nn.Conv2d(32,1,1)
        self.aux=nn.Conv2d(32,2,1) if bcd else None
    def forward(self,features,x):
        z=self.decoder(features,x)
        fg=self.fg(z)
        return fg if self.aux is None else torch.cat([fg,self.aux(z)],1)

class System(nn.Module):
    def __init__(self,enc,head):
        super().__init__()
        self.enc=enc.requires_grad_(False).eval()
        self.head=head
    def forward(self,x):
        self.enc.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            feat=self.enc.get_intermediate_layers(x.repeat(1,3,1,1),n=[5,11,17,23],reshape=True)
        assert len(feat)==4 and all(f.shape[-2:]==(16,16) for f in feat)
        return self.head(feat,x)

ARMS=('arm_a_multiscale_sem','arm_b_multiscale_bcd')

def build(arm):
    assert arm in ARMS
    net,p,digest=primus('Barrier100')
    torch.manual_seed(SEED)
    model=System(net.dino_encoder,Head(arm==ARMS[1]).cuda())
    return model,p,digest

def semantic(logit,y,valid):
    loss=F.binary_cross_entropy_with_logits(logit[valid],y[valid])
    p=logit.float().sigmoid()
    return loss+1-(2*(p*y*valid).sum()+1e-5)/((p*valid).sum()+(y*valid).sum()+1e-5)
