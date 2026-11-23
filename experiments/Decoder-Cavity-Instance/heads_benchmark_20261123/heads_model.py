from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / "decoder_cavity_inst_20261123"
sys.path.insert(0, str(PREVIOUS))
from model import ARMS, Conv, MultiScaleContextDecoder, System, build as build_reference, read

CONFIG = read(OUT / "config.json")


class ASPP(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = CONFIG["panoptic"]
        c = cfg["aspp_channels"]
        self.branches = nn.ModuleList([Conv(c, c, 1)])
        for dilation in cfg["dilations"]:
            self.branches.append(nn.Sequential(
                nn.Conv2d(c, c, 3, dilation=dilation, padding=dilation, groups=c, bias=False),
                Conv(c, c, 1),
            ))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), Conv(c, c, 1))
        self.project = nn.Sequential(Conv(c * 5, c, 1), nn.Dropout(cfg["dropout"]))

    def forward(self, z):
        pooled = self.pool(z).expand_as(z)
        return self.project(torch.cat([*(branch(z) for branch in self.branches), pooled], dim=1))


class PanopticBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.aspp = ASPP()
        self.decode = nn.ModuleList([Conv(256, 128), Conv(128, 64), Conv(96, 32), Conv(64, 32)])
        self.output = nn.Conv2d(32, channels, 1)
        if channels == 3:
            with torch.no_grad():
                self.output.bias[0].zero_()

    def forward(self, z, raw):
        z = self.aspp(z)
        for i, layer in enumerate(self.decode):
            z = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
            if i >= 2:
                z = torch.cat((z, F.avg_pool2d(raw, 2) if i == 2 else raw), dim=1)
            z = layer(z)
        return self.output(z)


class PanopticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.ModuleList(Conv(1024, 256, 1) for _ in range(4))
        self.raw = nn.Sequential(Conv(1, 16), Conv(16, 32))
        self.semantic = PanopticBranch(1)
        self.instances = PanopticBranch(3)

    def forward(self, features, image):
        z = torch.stack([p(f.float()) for p, f in zip(self.project, features)]).mean(0)
        raw = self.raw(image)
        return torch.cat((self.semantic(z, raw), self.instances(z, raw)), dim=1)


class FlowHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = MultiScaleContextDecoder()
        self.fg = nn.Conv2d(32, 1, 1)
        self.flow = nn.Conv2d(32, 2, 1)

    def forward(self, features, image):
        z = self.decoder(features, image)
        return torch.cat((self.fg(z), self.flow(z)), dim=1)


def build(arm):
    model, helper, digest = build_reference(ARMS[1] if arm == "batch32" else ARMS[0])
    if arm != "batch32":
        torch.manual_seed(CONFIG["seed"])
        head = PanopticHead() if arm == "panoptic" else FlowHead()
        model = System(model.enc, head.cuda())
    assert digest == CONFIG["encoder_sha256"]
    assert all(not p.requires_grad for p in model.enc.parameters())
    return model, helper


def load(arm):
    from model import sha

    record = read(OUT / arm / "trained.json")
    path = OUT / arm / "best.pt"
    assert record["weights_sha256"] == sha(path)
    model, _ = build(arm)
    model.head.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    return model.eval()
