"""CPU check against the published LAST-ViT selection equations."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from last_selection import install_ssl_hooks, select_patch_channels


def published_reference(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[-1]
    kernel = torch.exp(-0.5 * (torch.arange(-c // 2 + 1, c // 2 + 1).float() / math.sqrt(c)) ** 2)
    kernel = (kernel / torch.max(kernel)).to(x.device).view(1, 1, c)
    filtered = torch.fft.fft(x, dim=-1)
    filtered = torch.fft.fftshift(filtered, dim=-1) * kernel
    filtered = torch.fft.ifftshift(filtered, dim=-1)
    filtered = torch.fft.ifft(filtered, dim=-1).real
    score = x / torch.abs(filtered - x)
    indices = torch.topk(score, k=1, dim=1, largest=True).indices
    return torch.mean(torch.gather(x, 1, indices), dim=1)


def main() -> int:
    torch.manual_seed(260926)
    x = (torch.randn(3, 11, 1024) + 0.2).requires_grad_()
    actual, indices = select_patch_channels(x, return_indices=True)
    expected = published_reference(x)
    difference = float((actual - expected).abs().max())
    actual.square().mean().backward()
    gradient = x.grad
    class FakeBackbone(torch.nn.Module):
        def __init__(self, pair: bool):
            super().__init__()
            self.pair = pair

        def forward(self, patches: torch.Tensor):
            global_output = {"x_norm_clstoken": torch.full((patches.shape[0], patches.shape[-1]), 17.0),
                             "x_norm_patchtokens": patches}
            if not self.pair:
                return global_output
            local_output = {"x_norm_clstoken": torch.full((patches.shape[0], patches.shape[-1]), 19.0),
                            "x_norm_patchtokens": patches}
            return [global_output, local_output]

    class FakeNet:
        def __init__(self):
            self.student = type("Branch", (), {"backbone": FakeBackbone(True)})()
            self.model_ema = type("Branch", (), {"backbone": FakeBackbone(False)})()

    net = FakeNet()
    install_ssl_hooks(net)
    student_global, student_local = net.student.backbone(x.detach())
    teacher_global = net.model_ema.backbone(x.detach())
    local_preserved = bool(torch.all(student_local["x_norm_clstoken"] == 19))
    global_replaced = bool(torch.allclose(student_global["x_norm_clstoken"], actual.detach())
                           and torch.allclose(teacher_global["x_norm_clstoken"], actual.detach()))
    result = {
        "status": "complete",
        "shape": list(actual.shape),
        "indices_shape": list(indices.shape),
        "reference_max_abs_error": difference,
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "gradient_nonzero": int(torch.count_nonzero(gradient)) > 0,
        "selected_patch_indices_unique": int(indices.unique().numel()),
        "local_cls_preserved": local_preserved,
        "student_teacher_global_replaced": global_replaced,
    }
    assert difference <= 1e-6 and result["gradient_finite"] and result["gradient_nonzero"] and local_preserved and global_replaced, result
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
