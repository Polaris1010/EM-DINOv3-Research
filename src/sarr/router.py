"""Source-Adaptive Representation Routing (SARR) modules for EM mitochondria segmentation.

Contains:
  - SpatialRouter: Lightweight 2-layer 1x1 conv network predicting per-patch routing weights alpha.
  - SARR_Primus: Wraps two frozen DINOv3 ViT-L/16 encoders (P0 and B100k), the router,
    and the Primus PatchDecode segmentation head.
"""

from pathlib import Path
import math
import sys
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure segmentation directory is on path for official_primus imports
_SRC_DIR = Path(__file__).resolve().parents[1]
_SEG_DIR = _SRC_DIR / 'segmentation'
for _p in [str(_SRC_DIR), str(_SEG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Try importing upstream LayerNormNd and InitWeights_He; provide clean fallbacks
try:
    from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
except ImportError:
    class LayerNormNd(nn.Module):
        """Channels-first 2D LayerNorm: normalizes across channel dimension for (B, C, H, W)."""

        def __init__(self, channels: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.layer_norm(
                x.movedim(1, -1), (x.shape[1],), self.weight, self.bias, eps=self.eps
            ).movedim(-1, 1)

try:
    from dynamic_network_architectures.initialization.weight_init import InitWeights_He
except ImportError:
    class InitWeights_He:
        """He/Kaiming normal initialization for conv and conv transpose layers."""

        def __init__(self, neg_slope: float = 1e-2):
            self.neg_slope = neg_slope

        def __call__(self, module: nn.Module):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

try:
    from official_primus import PatchDecode
except ImportError:
    class PatchDecode(nn.Module):
        """Official Primus PatchDecode architecture upsampling 16x16 patch tokens to native 256x256."""

        def __init__(
            self,
            patch_size: int,
            embed_dim: int,
            out_channels: int,
            norm=LayerNormNd,
            activation=nn.GELU,
        ):
            super().__init__()
            assert patch_size > 0
            n = int(math.log2(patch_size))
            assert 2 ** n == patch_size and n >= 1

            ch = [embed_dim]
            for _ in range(n):
                ch.append(ch[-1] // 2)
            ch.append(out_channels)

            stages = []
            for i in range(n):
                stages.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(ch[i], ch[i + 1], kernel_size=2, stride=2),
                        norm(ch[i + 1]),
                        activation(),
                    )
                )
            stages.append(nn.Conv2d(ch[-2], ch[-1], kernel_size=1))
            self.decode = nn.Sequential(*stages)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decode(x)


class SpatialRouter(nn.Module):
    """Lightweight 2-layer 1x1 conv network for spatial representation routing.

    Architecture:
      - Input: Concatenation of P0 and B100k multi-scale features from 4 layers
               Shape: (B, 8192, H/16, W/16) for standard 4x1024-channel ViT-L features
      - Hidden: 256 channels, 1x1 conv + LayerNorm + GELU
      - Output: 1 channel, 1x1 conv + Sigmoid -> alpha in (0, 1) per patch

    Interpretation of alpha:
      - alpha -> 1.0 selects P0 (natural-image pretrained representation)
      - alpha -> 0.0 selects B100k (MitoVerse EM-continually pretrained representation)
      - Routed feature = alpha * f_P0 + (1 - alpha) * f_B100k
    """

    def __init__(
        self,
        in_channels: int = 8192,
        hidden_channels: int = 256,
        norm_layer=LayerNormNd,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.norm = norm_layer(hidden_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Weight initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        # Initialize conv2 with small weights and zero bias so routing begins at equal blend (alpha ~ 0.5)
        nn.init.normal_(self.conv2.weight, std=0.01)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-patch routing weights alpha.

        Args:
            x: Concatenated multi-encoder features of shape (B, in_channels, H_patch, W_patch).

        Returns:
            alpha: Routing weights of shape (B, 1, H_patch, W_patch) with values in (0, 1).
        """
        h = self.conv1(x)
        h = self.norm(h)
        h = self.act(h)
        alpha = self.sigmoid(self.conv2(h))
        return alpha


class SARR_Primus(nn.Module):
    """Source-Adaptive Representation Routing wrapper for frozen DINOv3 ViT-L encoders.

    Combines two frozen encoders (P0 and B100k), computes spatial routing weights,
    blends multi-scale intermediate layer representations, and decodes with PatchDecode.

    Supported modes:
      - 'sarr': Full 2-layer 1x1 conv spatial router predicting per-patch alpha in (0, 1) (default)
      - 'uniform_avg': Fixed alpha = 0.5 (equal blending, no learned routing)
      - 'global_gate': Single learned scalar parameter alpha shared across all spatial positions
      - 'p0_only': Pure P0 encoder baseline (alpha = 1.0, reproduces G-P0)
      - 'b100_only': Pure B100k encoder baseline (alpha = 0.0, reproduces G-B100)
    """

    def __init__(
        self,
        enc_p0: Optional[nn.Module] = None,
        enc_b100: Optional[nn.Module] = None,
        interaction_indices: List[int] = [5, 11, 17, 23],
        mode: str = 'sarr',
        embed_dim: int = 1024,
        patch_embed_size: int = 16,
        num_classes: int = 2,
        router_hidden_dim: int = 256,
        decoder_norm=None,
        decoder_act=nn.GELU,
    ):
        super().__init__()
        valid_modes = ('sarr', 'uniform_avg', 'global_gate', 'p0_only', 'b100_only')
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {valid_modes}")

        self.enc_p0 = enc_p0
        self.enc_b100 = enc_b100
        self.interaction_indices = list(interaction_indices)
        self.mode = mode
        self.embed_dim = embed_dim
        self.patch_embed_size = patch_embed_size
        self.num_classes = num_classes
        self.router_hidden_dim = router_hidden_dim

        # Number of intermediate layer outputs concatenated along channel dimension
        self.num_layers = len(self.interaction_indices)
        self.single_encoder_channels = self.embed_dim * self.num_layers  # 4 * 1024 = 4096

        # Freeze encoders: ensure no gradients and eval mode
        if self.enc_p0 is not None:
            self.enc_p0.requires_grad_(False)
            self.enc_p0.eval()
        if self.enc_b100 is not None:
            self.enc_b100.requires_grad_(False)
            self.enc_b100.eval()

        norm_layer = decoder_norm if decoder_norm is not None else LayerNormNd

        # PatchDecode upsamples from 4096 channels (blended intermediate features) to num_classes (2)
        self.up_projection = PatchDecode(
            patch_size=self.patch_embed_size,
            embed_dim=self.single_encoder_channels,
            out_channels=self.num_classes,
            norm=norm_layer,
            activation=decoder_act,
        )
        self.up_projection.apply(InitWeights_He(1e-2))

        # Routing mechanism setup
        if self.mode == 'sarr':
            self.router = SpatialRouter(
                in_channels=self.single_encoder_channels * 2,  # 4096 * 2 = 8192
                hidden_channels=self.router_hidden_dim,        # 256
                norm_layer=norm_layer,
            )
            self.gate = None
        elif self.mode == 'global_gate':
            # Single learned scalar parameter; initialized to 0.0 so sigmoid(0.0) = 0.5
            self.gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.router = None
        else:
            self.router = None
            self.gate = None

        # Cached routing weights from the most recent forward pass
        self.last_alpha: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        """Set training mode for trainable components while keeping encoders strictly in eval."""
        super().train(mode)
        if self.enc_p0 is not None:
            self.enc_p0.eval()
        if self.enc_b100 is not None:
            self.enc_b100.eval()
        return self

    def extract_encoder_features(
        self,
        encoder: nn.Module,
        x_rgb: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Extract multi-layer intermediate representations from a frozen encoder.

        Encoders are executed under torch.no_grad() and torch.autocast('cuda', dtype=torch.bfloat16)
        to minimize peak GPU VRAM footprint on a single 24GB RTX 4090.
        """
        encoder.eval()
        device_type = 'cuda' if x_rgb.is_cuda else 'cpu'
        with torch.no_grad():
            if device_type == 'cuda':
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    feats = encoder.get_intermediate_layers(
                        x_rgb, n=self.interaction_indices, reshape=True
                    )
            else:
                feats = encoder.get_intermediate_layers(
                    x_rgb, n=self.interaction_indices, reshape=True
                )
        return [f.float() for f in feats]

    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False,
        return_alpha: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run SARR forward pass.

        Args:
            x: Input image tensor of shape (B, 1, H, W). Grayscale is repeated 3 times for ViT.
            return_routing: If True, also returns the computed routing weight tensor alpha.
            return_alpha: Alias for return_routing.

        Returns:
            logits: Output segmentation logits of shape (B, num_classes, H, W).
            alpha: (Optional) Routing weights of shape (B, 1, H/16, W/16).
        """
        return_routing = return_routing or return_alpha
        assert x.dim() == 4 and x.shape[1] == 1, f"Expected (B, 1, H, W), got {tuple(x.shape)}"
        B, _, H, W = x.shape
        x_rgb = x.repeat(1, 3, 1, 1)
        h_patch = H // self.patch_embed_size
        w_patch = W // self.patch_embed_size

        if self.mode == 'p0_only':
            if self.enc_p0 is None:
                raise RuntimeError("enc_p0 is required for p0_only mode")
            f_p0 = self.extract_encoder_features(self.enc_p0, x_rgb)
            routed_features = torch.cat(f_p0, dim=1)
            alpha = torch.ones((B, 1, h_patch, w_patch), dtype=torch.float32, device=x.device)

        elif self.mode == 'b100_only':
            if self.enc_b100 is None:
                raise RuntimeError("enc_b100 is required for b100_only mode")
            f_b100 = self.extract_encoder_features(self.enc_b100, x_rgb)
            routed_features = torch.cat(f_b100, dim=1)
            alpha = torch.zeros((B, 1, h_patch, w_patch), dtype=torch.float32, device=x.device)

        else:
            if self.enc_p0 is None or self.enc_b100 is None:
                raise RuntimeError(f"Both enc_p0 and enc_b100 are required for mode '{self.mode}'")

            # Sequential extraction to avoid concurrent intermediate memory peaks
            f_p0 = self.extract_encoder_features(self.enc_p0, x_rgb)
            f_b100 = self.extract_encoder_features(self.enc_b100, x_rgb)

            f_p0_cat = torch.cat(f_p0, dim=1)      # (B, 4096, H/16, W/16)
            f_b100_cat = torch.cat(f_b100, dim=1)  # (B, 4096, H/16, W/16)

            if self.mode == 'sarr':
                router_in = torch.cat([f_p0_cat, f_b100_cat], dim=1)  # (B, 8192, H/16, W/16)
                alpha = self.router(router_in)                         # (B, 1, H/16, W/16)
            elif self.mode == 'uniform_avg':
                alpha = torch.full((B, 1, h_patch, w_patch), 0.5, dtype=torch.float32, device=x.device)
            elif self.mode == 'global_gate':
                scalar_alpha = torch.sigmoid(self.gate)
                alpha = scalar_alpha.view(1, 1, 1, 1).expand(B, 1, h_patch, w_patch)
            else:
                raise ValueError(f"Unhandled mode: {self.mode}")

            # Per-patch blended features: alpha * f_P0 + (1 - alpha) * f_B100k
            # Linearly equivalent to blending each layer and concatenating
            routed_features = alpha * f_p0_cat + (1.0 - alpha) * f_b100_cat

        self.last_alpha = alpha.detach()
        logits = self.up_projection(routed_features)

        if return_routing:
            return logits, alpha
        return logits
