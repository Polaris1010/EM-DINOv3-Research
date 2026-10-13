"""Native ViT-L patch wrapper with the existing four-scale PatchDecode layout."""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class LayerNormNd(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return F.layer_norm(x.movedim(1, -1), (x.shape[1],), self.weight, self.bias).movedim(-1, 1)


class PatchDecode(nn.Module):
    """Same 4-stage transposed-convolution topology as archived Primus PatchDecode."""

    def __init__(self):
        super().__init__()
        channels = (4096, 2048, 1024, 512, 256)
        stages = []
        for a, b in zip(channels[:-1], channels[1:]):
            stages.append(nn.Sequential(nn.ConvTranspose2d(a, b, 2, stride=2),
                                        LayerNormNd(b), nn.GELU()))
        stages.append(nn.Conv2d(256, 2, 1))
        self.decode = nn.Sequential(*stages)
        self.apply(self._he_init)

    @staticmethod
    def _he_init(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.decode(x)


def load_encoder(dinov3_root: Path, weight: Path):
    import sys
    sys.path.insert(0, str(dinov3_root))
    from dinov3.models.vision_transformer import vit_large
    checkpoint = torch.load(weight, map_location='cpu', mmap=True, weights_only=False)
    if 'teacher' in checkpoint:
        state = {key.removeprefix('backbone.'): value
                 for key, value in checkpoint['teacher'].items() if key.startswith('backbone.')}
    else:
        state = checkpoint
    with torch.device('meta'):
        encoder = vit_large(patch_size=16, n_storage_tokens=4, mask_k_bias=True,
                            layerscale_init=1e-5, pos_embed_rope_base=100.0,
                            pos_embed_rope_normalize_coords='separate',
                            pos_embed_rope_dtype='bf16')
    encoder.load_state_dict(state, strict=True, assign=True)
    return encoder


class SegModel(nn.Module):
    def __init__(self, encoder: nn.Module, head_seed: int, fine_tune_last4: bool):
        super().__init__()
        self.encoder = encoder
        self.fine_tune_last4 = fine_tune_last4
        self.encoder.requires_grad_(False)
        if fine_tune_last4:
            for block in self.encoder.blocks[20:24]:
                block.requires_grad_(True)
        torch.manual_seed(20_000 + head_seed)
        self.decoder = PatchDecode()
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (3, 256, 256):
            raise ValueError(f'Expected [B,3,256,256], got {tuple(x.shape)}')
        grad_scope = contextlib.nullcontext() if self.fine_tune_last4 else torch.no_grad()
        with grad_scope:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
                features = self.encoder.get_intermediate_layers(x, n=[5, 11, 17, 23], reshape=True)
        if len(features) != 4 or any(f.shape[1:] != (1024, 16, 16) for f in features):
            raise RuntimeError('Special-token removal or patch-grid mismatch')
        return self.decoder(torch.cat([f.float() for f in features], dim=1))
