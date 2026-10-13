"""LAST-inspired per-channel patch selection for DINOv3 global SSL heads.

The reference LAST-ViT implementation uses a Gaussian kernel of width C and
sigma sqrt(C), FFT over the feature channel, then top-1 over patches for each
channel. Its ViT-B uses C=768; this adaptation uses the native ViT-L C=1024.
"""

from __future__ import annotations

import math

import torch


def select_patch_channels(patch_tokens: torch.Tensor, *, return_indices: bool = False):
    if patch_tokens.ndim != 3 or patch_tokens.shape[1] < 1:
        raise ValueError("Expected patch tokens [batch, patches, channels]")
    original_dtype = patch_tokens.dtype
    patches = patch_tokens.float()
    channels = patches.shape[-1]
    coord = torch.arange(-channels // 2 + 1, channels // 2 + 1, device=patches.device,
                         dtype=torch.float32)
    if coord.numel() != channels:
        raise AssertionError("Gaussian kernel length differs from channel count")
    kernel = torch.exp(-0.5 * (coord / math.sqrt(channels)).square())
    kernel = (kernel / kernel.max()).view(1, 1, channels)
    filtered = torch.fft.fft(patches, dim=-1)
    filtered = torch.fft.fftshift(filtered, dim=-1) * kernel
    filtered = torch.fft.ifftshift(filtered, dim=-1)
    filtered = torch.fft.ifft(filtered, dim=-1).real
    scores = patches / (filtered - patches).abs().clamp_min(1e-6)
    indices = torch.topk(scores, k=1, dim=1, largest=True).indices
    selected = torch.gather(patches, 1, indices).mean(dim=1).to(original_dtype)
    if return_indices:
        return selected, indices.squeeze(1)
    return selected


def install_ssl_hooks(net) -> None:
    """Replace global CLS only in student and EMA; preserve local CLS and patches.

    KoLeo and the DINO global head read the replacement from the existing
    SSLMetaArch output. The hooks have no state-dict keys and must be
    reinstalled on checkpoint load.
    """

    def replace_student(_module, _args, result):
        global_output = result[0] if isinstance(result, list) else result
        global_output["x_norm_clstoken"] = select_patch_channels(global_output["x_norm_patchtokens"])
        return result

    def replace_teacher(_module, _args, result):
        result["x_norm_clstoken"] = select_patch_channels(result["x_norm_patchtokens"])
        return result

    net.student.backbone.register_forward_hook(replace_student)
    net.model_ema.backbone.register_forward_hook(replace_teacher)
