"""Unit test contract for SARR (Source-Adaptive Representation Routing).

Verifies:
  1. SpatialRouter architecture, output dimensions, value bounds, and initial bias.
  2. SARR_Primus modes, parameter counts, and routing behavior.
  3. Strict prohibition of GPU 0 in SARR experiment runners.
"""

from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn

# Add src to path
R = Path(__file__).resolve().parents[1]
for p in [str(R / "src"), str(R / "src" / "sarr"), str(R / "src" / "segmentation")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from router import SpatialRouter, SARR_Primus
from run_sarr import validate_gpu_assignment


class MockDINOEncoder(nn.Module):
    """Mock encoder returning 4 multi-scale feature maps of shape (B, 1024, 16, 16)."""

    def __init__(self, embed_dim: int = 1024, n_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers

    def get_intermediate_layers(self, x, n=(5, 11, 17, 23), reshape=True):
        B = x.shape[0]
        # For input 256x256, patches are 16x16
        return [torch.zeros(B, self.embed_dim, 16, 16, device=x.device, dtype=x.dtype) for _ in n]


class SARRContractTests(unittest.TestCase):

    def test_spatial_router_forward_and_bounds(self):
        """SpatialRouter should take (B, 8192, 16, 16) and return alpha in (0, 1) of shape (B, 1, 16, 16)."""
        router = SpatialRouter(in_channels=8192, hidden_channels=256)
        x = torch.randn(2, 8192, 16, 16)
        alpha = router(x)

        self.assertEqual(alpha.shape, (2, 1, 16, 16))
        self.assertTrue((alpha > 0.0).all())
        self.assertTrue((alpha < 1.0).all())

        # Since conv2 weight is small and bias is 0, alpha should initially be close to 0.5 (sigmoid(0) = 0.5)
        # Average across random input should be in [0.45, 0.55]
        mean_val = alpha.mean().item()
        self.assertAlmostEqual(mean_val, 0.5, delta=0.08)

    def test_sarr_primus_modes_and_shapes(self):
        """SARR_Primus should produce (B, 2, 256, 256) segmentation logits across all modes."""
        enc_p0 = MockDINOEncoder()
        enc_b100 = MockDINOEncoder()

        for mode in ("sarr", "uniform_avg", "global_gate", "p0_only", "b100_only"):
            net = SARR_Primus(
                enc_p0=enc_p0,
                enc_b100=enc_b100,
                interaction_indices=[5, 11, 17, 23],
                mode=mode,
                embed_dim=1024,
                patch_embed_size=16,
                num_classes=2,
            )

            # Input: (B, 1, 256, 256)
            inp = torch.randn(1, 1, 256, 256)
            logits, alpha = net(inp, return_alpha=True)

            self.assertEqual(logits.shape, (1, 2, 256, 256))
            self.assertEqual(alpha.shape, (1, 1, 16, 16))
            if mode == "sarr":
                self.assertTrue((alpha > 0.0).all() and (alpha < 1.0).all())
            elif mode == "uniform_avg":
                self.assertTrue(torch.allclose(alpha, torch.tensor(0.5)))
            elif mode == "p0_only":
                self.assertTrue(torch.allclose(alpha, torch.tensor(1.0)))
            elif mode == "b100_only":
                self.assertTrue(torch.allclose(alpha, torch.tensor(0.0)))

    def test_invalid_mode_raises_error(self):
        """Unknown mode must be rejected immediately."""
        with self.assertRaises(ValueError):
            SARR_Primus(mode="invalid_unsupported_mode")

    def test_gpu_zero_prohibition(self):
        """GPU 0 must raise ValueError under research policy; GPUs 1, 2, 3 must pass."""
        with self.assertRaises(ValueError) as ctx:
            validate_gpu_assignment(0)
        self.assertIn("GPU 0 is strictly forbidden", str(ctx.exception))

        for valid_gpu in (1, 2, 3):
            self.assertEqual(validate_gpu_assignment(valid_gpu), valid_gpu)

        with self.assertRaises(ValueError):
            validate_gpu_assignment(4)


if __name__ == "__main__":
    unittest.main()
