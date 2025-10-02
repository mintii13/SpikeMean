import copy
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn

from spikingjelly.activation_based import neuron, functional

__all__ = ["SpikeDrivenUniAD"]


# ============================================================
# SPIKE-DRIVEN SELF-ATTENTION (SDSA)
# ============================================================
class SpikeDrivenSelfAttention(nn.Module):
    """
    Spike-Driven Self-Attention theo BICLab implementation
    Sử dụng spikingjelly 0.0.0.0.14
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Q, K, V projections (Conv1x1)
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='s')
        
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='s')
        
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='s')
        
        # Attention LIF (threshold = 0.5 theo paper)
        self.attn_lif = neuron.LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, step_mode='s')
        
        # Output projection
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)
        
        # Shortcut LIF (membrane shortcut)
        self.shortcut_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='s')
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - input feature map
        Returns:
            output: [B, C, H, W] - attention output
        """
        B, C, H, W = x.shape
        N = H * W
        identity = x
        
        # Membrane shortcut: LIF before processing
        x = self.shortcut_lif(x)
        
        # Generate Q, K, V
        q = self.q_conv(x)
        q = self.q_bn(q)
        q = self.q_lif(q)  # Binary spikes
        q = q.flatten(2).transpose(1, 2)  # [B, N, C]
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        
        k = self.k_conv(x)
        k = self.k_bn(k)
        k = self.k_lif(k)  # Binary spikes
        k = k.flatten(2).transpose(1, 2)  # [B, N, C]
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        
        v = self.v_conv(x)
        v = self.v_bn(v)
        v = self.v_lif(v)  # Binary spikes
        v = v.flatten(2).transpose(1, 2)  # [B, N, C]
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        
        # Spike-Driven Attention: kv = k * v (element-wise mask)
        kv = k.mul(v)  # [B, H, N, D] - Hadamard product (mask operation)
        kv = kv.sum(dim=2, keepdim=True)  # [B, H, 1, D] - column sum
        kv = self.attn_lif(kv)  # Binary attention vector
        
        # Apply attention: x = q * kv
        x = q.mul(kv)  # [B, H, N, D] - mask operation
        
        # Reshape back
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        # Output projection
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        
        # Residual connection (membrane shortcut)
        x = x + identity
        
        return x


# ============================================================
# SPIKE-DRIVEN MLP
# ============================================================
class SpikeDrivenMLP(nn.Module):
    """
    Spike-Driven MLP theo BICLab implementation
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = (in_features == hidden_features)
        
        # First FC layer
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='s')
        
        # Second FC layer
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='s')
        
        self.c_hidden = hidden_features
        self.c_output = out_features
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            output: [B, C, H, W]
        """
        identity = x
        
        # First layer with membrane shortcut
        x = self.fc1_lif(x)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x)
        
        if self.res:
            x = identity + x
            identity = x
        
        # Second layer with membrane shortcut
        x = self.fc2_lif(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x)
        
        # Residual
        x = x + identity
        return x


# ============================================================
# SPIKE-DRIVEN ENCODER/DECODER BLOCK
# ============================================================
class SpikeDrivenBlock(nn.Module):
    """
    Spike-Driven Transformer Block
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        
        # Spike-Driven Self-Attention
        self.attn = SpikeDrivenSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        # Spike-Driven MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SpikeDrivenMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            output: [B, C, H, W]
        """
        # Attention block (with internal residual)
        x = self.attn(x)
        
        # MLP block (with internal residual)
        x = self.mlp(x)
        
        return x


# ============================================================
# SPIKE-DRIVEN UniAD
# ============================================================
class SpikeDrivenUniAD(nn.Module):
    def __init__(
        self,
        inplanes,
        instrides,
        feature_size,
        feature_jitter,
        neighbor_mask,
        hidden_dim,
        pos_embed_type,
        save_recon,
        initializer,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1
        
        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]
        self.feature_jitter = feature_jitter
        self.save_recon = save_recon
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Conv2d(inplanes[0], hidden_dim, kernel_size=1)
        self.input_bn = nn.BatchNorm2d(hidden_dim)
        self.input_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='s')
        
        # Spike-Driven Encoder
        num_encoder_layers = kwargs.get('num_encoder_layers', 4)
        self.encoder = nn.ModuleList([
            SpikeDrivenBlock(
                dim=hidden_dim,
                num_heads=kwargs.get('nhead', 8),
                mlp_ratio=kwargs.get('dim_feedforward', 1024) / hidden_dim,
                qkv_bias=False,
                drop=kwargs.get('dropout', 0.1),
                attn_drop=kwargs.get('dropout', 0.1),
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Spike-Driven Decoder
        num_decoder_layers = kwargs.get('num_decoder_layers', 4)
        self.decoder = nn.ModuleList([
            SpikeDrivenBlock(
                dim=hidden_dim,
                num_heads=kwargs.get('nhead', 8),
                mlp_ratio=kwargs.get('dim_feedforward', 1024) / hidden_dim,
                qkv_bias=False,
                drop=kwargs.get('dropout', 0.1),
                attn_drop=kwargs.get('dropout', 0.1),
            )
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, inplanes[0], kernel_size=1)
        
        # Upsampling
        self.upsample_scale = instrides[0]
        
        # Initialize
        initialize_from_cfg(self, initializer)
        
    def forward(self, input):
        # Get features from backbone
        if "feature_align" in input:
            features = input["feature_align"]
        else:
            features = input["features"][-1]
        
        # Project input
        x = self.input_proj(features)  # [B, C, H, W]
        x = self.input_bn(x)
        x = self.input_lif(x)  # Convert to spikes
        
        # Reset neuron states at start
        functional.reset_net(self)
        
        # Encode
        for encoder_block in self.encoder:
            x = encoder_block(x)
        
        # Decode
        for decoder_block in self.decoder:
            x = decoder_block(x)
        
        # Output
        feature_rec = self.output_proj(x)
        feature_rec = torch.sigmoid(feature_rec)
        
        # Compute prediction (reconstruction error)
        if "feature_align" in input:
            feature_compare = input["feature_align"]
        else:
            feature_compare = input["features"][-1]
        
        feature_compare = torch.sigmoid(feature_compare)
        pred = torch.sqrt(
            torch.sum((feature_rec - feature_compare) ** 2, dim=1, keepdim=True)
        )
        
        # Upsample
        pred = F.interpolate(
            pred,
            scale_factor=self.upsample_scale,
            mode='bicubic',
            align_corners=False
        )
        
        return {
            "feature_rec": feature_rec,
            "feature_align": feature_compare,
            "pred": pred,
        }


# Position embedding (giữ nguyên)
def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    if pos_embed_type in ("v2", "sine"):
        pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"):
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed


class PositionEmbeddingSine(nn.Module):
    def __init__(self, feature_size, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]), device=tensor.device)
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)
        return pos


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)
        j = torch.arange(self.feature_size[0], device=tensor.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            torch.cat([x_emb.unsqueeze(0)] * self.feature_size[0], dim=0),
            torch.cat([y_emb.unsqueeze(1)] * self.feature_size[1], dim=1),
        ], dim=-1).flatten(0, 1)
        return pos