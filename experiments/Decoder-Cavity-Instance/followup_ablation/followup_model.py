from pathlib import Path
import sys

import torch
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
sys.path.insert(0, str(PARENT))

from model import ARMS as PARENT_ARMS
from model import BASE, Conv, SEED, atomic, build as build_parent, gpu_guard, read, semantic, sha

ARMS = ("arm_a_norawskip",)


def build(arm):
    assert arm == ARMS[0]
    model, metadata, encoder_sha = build_parent(PARENT_ARMS[0])
    head = model.head
    original = read(PARENT / PARENT_ARMS[0] / "train_config.json")
    assert metadata.state_hash(head.state_dict()) == original["initial_shared_head_sha"]
    assert encoder_sha == original["encoder_sha"]
    decoder = head.decoder
    for i in (2, 3):
        old = decoder.up[i]
        conv = old[0]
        new = Conv(conv.in_channels - 32, conv.out_channels).to(conv.weight)
        with torch.no_grad():
            new[0].weight.copy_(conv.weight[:, :new[0].in_channels])
            new[1].load_state_dict(old[1].state_dict())
        decoder.up[i] = new
    del decoder.raw
    decoder.forward = forward_without_raw.__get__(decoder)
    return model, metadata, encoder_sha


def forward_without_raw(self, features, image):
    x = self.context(torch.stack([p(f.float()) for p, f in zip(self.project, features)]).mean(0))
    for layer in self.up:
        x = layer(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
    return x


def load():
    model, _, encoder_sha = build(ARMS[0])
    directory = OUT / ARMS[0]
    record = read(directory / "trained.json")
    path = directory / "model_40000.pt"
    assert encoder_sha == record["config"]["encoder_sha"]
    assert record["step"] == 40000 and sha(path) == record["model_sha256"]
    model.head.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    return model.eval(), record
