import torch
from einops import rearrange
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)
from typing import Tuple
import torch
from torch import nn

from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from einops import rearrange
import numpy as np

import math
class LaSt_FrequencyGate(nn.Module):
    """
    稳定版 LAST-ViT 启发的通道感知频域门控模块。
    用于DINOv3多层特征的自适应过滤，避免原始除法形式导致NaN。
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # 可学习平滑尺度
        self.sigma = nn.Parameter(torch.ones(1) * (embed_dim ** 0.5))

        self.channel_projector = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1, bias=False),
            LayerNormNd(embed_dim // 4),
            nn.GELU(),

            nn.Conv2d(
                embed_dim // 4,
                embed_dim // 4,
                kernel_size=3,
                padding=1,
                groups=embed_dim // 4,
                bias=False
            ),
            LayerNormNd(embed_dim // 4),
            nn.GELU(),

            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # 不要从0开始，也不要太大；1e-3更稳
        self.gamma = nn.Parameter(torch.ones(1) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 保证输入本身没有NaN或Inf
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]

        # 动态生成高斯核
        sigma = torch.clamp(self.sigma, min=1.0, max=float(C))
        freq_index = torch.arange(
            -C // 2 + 1,
            C // 2 + 1,
            device=x.device
        ).float()

        kernel = torch.exp(-0.5 * (freq_index / sigma) ** 2)
        kernel = kernel / (kernel.max() + 1e-6)
        gs_k = kernel.view(1, 1, C)

        # 通道维FFT平滑
        x_fft = torch.fft.fft(x_flat, dim=-1)
        x_fft = torch.fft.fftshift(x_fft, dim=-1)
        x_fft = x_fft * gs_k
        x_fft = torch.fft.ifftshift(x_fft, dim=-1)
        x_smooth = torch.fft.ifft(x_fft, dim=-1).real

        # 稳定版显著性差异：不用 x / abs(x_smooth - x)
        residual = x_flat - x_smooth

        # 对残差做标准化，避免数值过大
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = residual.std(dim=-1, keepdim=True) + 1e-6
        diff = (residual - residual_mean) / residual_std

        # 限幅，防止进入后续卷积前过大
        diff = torch.clamp(diff, min=-5.0, max=5.0)
        diff = torch.nan_to_num(diff, nan=0.0, posinf=5.0, neginf=-5.0)

        diff_spatial = diff.permute(0, 2, 1).view(B, C, H, W)

        gate = self.channel_projector(diff_spatial)
        gate = torch.nan_to_num(gate, nan=0.5, posinf=1.0, neginf=0.0)

        # 限制gamma，防止残差门控放大过强
        gamma = torch.clamp(self.gamma, min=0.0, max=0.1)

        return x + gamma * (x * gate)
    
class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
        self,
        patch_size: int, 
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()
        assert patch_size > 0
        n = int(math.log2(patch_size))

        assert 2 ** n == patch_size and n >= 1

        ch = [embed_dim]
        for _ in range(n):
            ch.append(ch[-1]//2)
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

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py)! This will require you to reshape the output of your transformer!
        """
        return self.decode(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = False

class Primus(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1
        x = x.repeat(1,3,1,1)
        x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        dec_out = self.up_projection(x)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
    

class Primus_Multiscale(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        interaction_indices =[1,2,3,4]
    ):
        """
        We follow a similar design as ViT-adapter, using intermediate layers and concat along channel dimension.
        """
        super().__init__()

        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim * len(interaction_indices), num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        self.decoder = Decoder()
        self.up_projection.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices
        # 【新增】为每一层提取出的特征准备一个独立的 LaSt 过滤器
        self.last_gates = nn.ModuleList([
            LaSt_FrequencyGate(embed_dim) for _ in range(len(interaction_indices))
        ])

    def forward(self, x, ret_mask=False):
        assert x.shape[1] == 1

        x = x.repeat(1, 3, 1, 1)

        hier = self.dino_encoder.get_intermediate_layers(
            x,
            n=self.interaction_indices,
            reshape=True
        )

        filtered_hier = []
        for feat, gate_layer in zip(hier, self.last_gates):
            filtered_hier.append(gate_layer(feat))

        # 这里一定用 filtered_hier
        hier = torch.cat(filtered_hier, dim=1)

        dec_out = self.up_projection(hier)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")