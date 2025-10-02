import copy
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn

__all__ = ["UniADMemory"]


class ChannelMemoryModule(nn.Module):
    """
    Channel Memory Module - Feature-wise processing với Q/K/V projections
    Memory shape: [mem_size, feature_dim] - học patterns trong feature space
    """
    def __init__(self, mem_dim, feature_dim, **kwargs):
        super(ChannelMemoryModule, self).__init__()
        
        self.mem_dim = mem_dim  # Number of memory slots
        self.feature_dim = feature_dim  # Feature dimension
        self.scale = 1.0 / math.sqrt(feature_dim)  # Scale factor for attention
        
        # Memory bank: [mem_size, feature_dim]
        self.memory = nn.Parameter(torch.randn(mem_dim, feature_dim))
        nn.init.normal_(self.memory, mean=0, std=0.1)
        
        # Q, K, V projections như trong attention mechanism
        self.query_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        
    def forward(self, input_tokens):
        """
        Args:
            input_tokens: [N_tokens, batch_size, feature_dim] - feature tokens
        Returns:
            dict with output and memory information
        """
        N_tokens, batch_size, feature_dim = input_tokens.shape
        
        # Reshape for processing: [N_tokens * batch_size, feature_dim]
        input_flat = input_tokens.view(N_tokens * batch_size, feature_dim)
        
        # Project input to queries
        queries = self.query_proj(input_flat)  # [N_tokens * batch_size, feature_dim]
        
        # Project memory to keys and values
        keys = self.key_proj(self.memory)  # [mem_dim, feature_dim]
        values = self.value_proj(self.memory)  # [mem_dim, feature_dim]
        
        # Compute attention scores: Q @ K^T
        attention_scores = torch.mm(queries, keys.t())  # [N_tokens * batch_size, mem_dim]
        
        # Apply scale
        attention_scores = attention_scores * self.scale
        
        # Apply softmax to get attention weights
        att_weight = F.softmax(attention_scores, dim=1)  # [N_tokens * batch_size, mem_dim]
        
        # Retrieve from memory: attention × values
        output_flat = torch.mm(att_weight, values)  # [N_tokens * batch_size, feature_dim]
        
        # Reshape back to original format
        output_tokens = output_flat.view(N_tokens, batch_size, feature_dim)  # [N_tokens, batch_size, feature_dim]
        
        return {
            'output': output_tokens,
            'att_weight': att_weight,
            'attention_scores': attention_scores,
            'memory': self.memory
        }


class SpatialMemoryModule(nn.Module):
    """
    Spatial Memory Module - Spatial pattern processing với Q/K/V projections và SSIM similarity
    Memory shape: [mem_size, H, W] - học spatial patterns
    """
    def __init__(self, mem_dim, height, width, **kwargs):
        super(SpatialMemoryModule, self).__init__()
        
        self.mem_dim = mem_dim  # Number of memory slots
        self.height = height    # Spatial height
        self.width = width      # Spatial width
        self.spatial_dim = height * width
        self.scale = 1.0  # Scale factor for SSIM similarity
        
        # Memory bank: [mem_size, H, W] - giữ nguyên spatial structure
        self.memory = nn.Parameter(torch.randn(mem_dim, height, width))
        nn.init.normal_(self.memory, mean=0, std=0.1)
        
        # Q, K, V projections cho spatial patterns
        self.query_proj = nn.Linear(self.spatial_dim, self.spatial_dim, bias=False)
        self.key_proj = nn.Linear(self.spatial_dim, self.spatial_dim, bias=False)
        self.value_proj = nn.Linear(self.spatial_dim, self.spatial_dim, bias=False)

    def compute_ssim_similarity(self, query_patterns, memory_patterns):
        """
        Compute SSIM similarity between query spatial patterns and memory patterns
        
        Args:
            query_patterns: [N_patterns, H, W] - query spatial patterns
            memory_patterns: [mem_dim, H, W] - memory spatial patterns
        Returns:
            similarity: [N_patterns, mem_dim] - SSIM similarities
        """
        N_patterns, H, W = query_patterns.shape
        mem_dim = memory_patterns.shape[0]
        
        # Flatten spatial dimensions
        query_flat = query_patterns.view(N_patterns, H * W)  # [N_patterns, H*W]
        memory_flat = memory_patterns.view(mem_dim, H * W)  # [mem_dim, H*W]
        
        # Compute means
        query_mean = torch.mean(query_flat, dim=1, keepdim=True)  # [N_patterns, 1]
        memory_mean = torch.mean(memory_flat, dim=1, keepdim=True)  # [mem_dim, 1]
        
        # Compute variances
        query_var = torch.var(query_flat, dim=1, keepdim=True)  # [N_patterns, 1]
        memory_var = torch.var(memory_flat, dim=1, keepdim=True)  # [mem_dim, 1]
        
        # Center the data
        query_centered = query_flat - query_mean
        memory_centered = memory_flat - memory_mean
        
        # Compute covariance
        covariance = torch.mm(query_centered, memory_centered.t()) / (H * W - 1)  # [N_patterns, mem_dim]
        
        # SSIM formula components
        c1, c2 = 0.01, 0.03
        
        # Numerator: (2*mu1*mu2 + c1) * (2*cov + c2)
        mean_product = torch.mm(query_mean, memory_mean.t())  # [N_patterns, mem_dim]
        numerator = (2 * mean_product + c1) * (2 * covariance + c2)
        
        # Denominator: (mu1^2 + mu2^2 + c1) * (var1 + var2 + c2)
        mean_sum = query_mean**2 + memory_mean.t()**2  # [N_patterns, mem_dim]
        var_sum = query_var + memory_var.t()  # [N_patterns, mem_dim]
        denominator = (mean_sum + c1) * (var_sum + c2)
        
        # SSIM similarity
        ssim = numerator / (denominator + 1e-8)
        
        return ssim

    def forward(self, input_tokens):
        """
        Args:
            input_tokens: [N_tokens, batch_size, feature_dim] - feature tokens
        Returns:
            dict with output and memory information
        """
        feature_dim, batch_size, N_tokens = input_tokens.shape
        
        # Reshape để có spatial dimensions (giả sử feature_dim = height * width)
        H, W = self.height, self.width
        
        # Reshape to spatial format: [N_tokens * batch_size, H, W]
        input_spatial = input_tokens.view(N_tokens * batch_size, H, W)
        
        # Project input to queries
        input_flat = input_spatial.view(N_tokens * batch_size, H * W)  # [N_tokens * batch_size, H*W]
        queries_flat = self.query_proj(input_flat)  # [N_tokens * batch_size, H*W]
        queries_spatial = queries_flat.view(N_tokens * batch_size, H, W)  # [N_tokens * batch_size, H, W]
        
        # Project memory to keys and values
        memory_flat = self.memory.view(self.mem_dim, H * W)  # [mem_dim, H*W]
        keys_flat = self.key_proj(memory_flat)  # [mem_dim, H*W]
        values_flat = self.value_proj(memory_flat)  # [mem_dim, H*W]
        keys_spatial = keys_flat.view(self.mem_dim, H, W)  # [mem_dim, H, W]
        
        # Compute SSIM similarity between queries và keys
        ssim_similarity = self.compute_ssim_similarity(queries_spatial, keys_spatial)  # [N_tokens * batch_size, mem_dim]
        
        # Apply scale and softmax to get attention weights
        attention_scores = ssim_similarity * self.scale
        att_weight = F.softmax(attention_scores, dim=1)  # [N_tokens * batch_size, mem_dim]
        
        # Retrieve from memory: attention × values
        output_flat = torch.mm(att_weight, values_flat)  # [N_tokens * batch_size, H*W]
        
        # Reshape back to original token format
        output_tokens = output_flat.view(N_tokens, batch_size, feature_dim)  # [N_tokens, batch_size, feature_dim]
        
        return {
            'output': output_tokens,
            'att_weight': att_weight,
            'ssim_similarity': ssim_similarity,
            'memory': self.memory
        }


class UniADMemory(nn.Module):
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
        self.pos_embed = build_position_embedding(
            pos_embed_type, feature_size, hidden_dim
        )
        self.save_recon = save_recon
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(inplanes[0], hidden_dim)
        
        # Memory modules configuration
        self.memory_mode = kwargs.get('memory_mode', 'both')  # 'channel', 'spatial', 'both', 'none'
        self.channel_memory_size = kwargs.get('channel_memory_size', 256)
        self.spatial_memory_size = kwargs.get('spatial_memory_size', 256)
        
        # Initialize memory modules based on mode
        self.use_channel_memory = self.memory_mode in ['channel', 'both']
        self.use_spatial_memory = self.memory_mode in ['spatial', 'both']
        
        self.channel_memory_module = None
        self.spatial_memory_module = None
        
        if self.use_channel_memory:
            # Channel memory module - xử lý feature patterns
            self.channel_memory_module = ChannelMemoryModule(
                mem_dim=self.channel_memory_size,
                feature_dim=hidden_dim,
                **kwargs
            )
            print('use channel mem')
        else:
            print('no channel mem')
        
        if self.use_spatial_memory:
            # Spatial memory module - xử lý spatial patterns  
            self.spatial_memory_module = SpatialMemoryModule(
                mem_dim=self.spatial_memory_size,
                height=feature_size[0],
                width=feature_size[1],
                **kwargs
            )
            print('use spatial mem')
        else:
            print('no spatial mem')
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            kwargs.get('nhead', 8), 
            kwargs.get('dim_feedforward', 1024),
            kwargs.get('dropout', 0.1),
            kwargs.get('activation', 'relu'),
            kwargs.get('normalize_before', False)
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if kwargs.get('normalize_before', False) else None
        self.encoder = TransformerEncoder(
            encoder_layer, 
            kwargs.get('num_encoder_layers', 4),
            encoder_norm
        )
        
        # Decoder
        decoder_layer = TransformerMemoryDecoderLayer(
            hidden_dim,
            kwargs.get('nhead', 8),
            kwargs.get('dim_feedforward', 1024),
            kwargs.get('dropout', 0.1),
            kwargs.get('activation', 'relu'),
            kwargs.get('normalize_before', False),
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            kwargs.get('num_decoder_layers', 4),
            decoder_norm,
            return_intermediate=False,
        )
        
        # Feature fusion layer - adapt based on memory mode
        fusion_input_dim = hidden_dim
        if self.use_channel_memory and self.use_spatial_memory:
            fusion_input_dim = hidden_dim * 2
        elif self.use_channel_memory or self.use_spatial_memory:
            fusion_input_dim = hidden_dim
        
        self.fusion_layer = nn.Linear(fusion_input_dim, hidden_dim) if fusion_input_dim > hidden_dim else nn.Identity()
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, inplanes[0])
        
        # Upsampling
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])

        # Initialize parameters
        initialize_from_cfg(self, initializer)

    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).to(feature_tokens.device)
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward(self, input):
        feature_align = input["feature_align"]  # B x C X H x W
        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"
        )  # (H x W) x B x C
        
        # Add jitter during training if enabled
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )
            
        # Project input features
        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C
        feature_tokens = F.layer_norm(feature_tokens, feature_tokens.shape[-1:])
        # Get positional embeddings
        pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C
        
        # Encode features using transformer encoder
        encoded_tokens = self.encoder(
            feature_tokens, pos=pos_embed
        )  # (H x W) x B x C
        
        # Memory retrieval based on memory mode
        memory_features_list = []
        channel_result = None
        spatial_result = None
        
        if self.use_channel_memory:
            channel_result = self.channel_memory_module(encoded_tokens)
            channel_retrieved = channel_result['output']  # (H x W) x B x C
            memory_features_list.append(channel_retrieved)
        
        if self.use_spatial_memory:
            spatial_result = self.spatial_memory_module(encoded_tokens)
            spatial_retrieved = spatial_result['output']  # C x B x H x W
            spatial_retrieved = torch.permute(spatial_retrieved, (2, 1, 0))  # (H x W) x B x C
            memory_features_list.append(spatial_retrieved)
        
        # Fuse memory features based on available memories
        if len(memory_features_list) == 0:
            # No memory - use encoded tokens directly
            memory_features = encoded_tokens
        elif len(memory_features_list) == 1:
            # Single memory type
            memory_features = self.fusion_layer(memory_features_list[0])
        else:
            # Multiple memory types - concatenate and fuse
            combined_features = torch.cat(memory_features_list, dim=-1)  # (H x W) x B x (N*C)
            memory_features = self.fusion_layer(combined_features)  # (H x W) x B x C
        
        # Decode features
        decoded_tokens = self.decoder(
            memory_features, 
            encoded_tokens, 
            pos=pos_embed
        )  # (H x W) x B x C
        
        # Project back to original dimension
        feature_rec_tokens = self.output_proj(decoded_tokens)  # (H x W) x B x C
        feature_rec_tokens = torch.sigmoid(feature_rec_tokens)
        # Reshape back to spatial representation
        feature_rec = rearrange(
            feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W

        # Save reconstructed features if needed
        if not self.training and self.save_recon:
            clsnames = input["clsname"]
            filenames = input["filename"]
            for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
                filedir, filename = os.path.split(filename)
                _, defename = os.path.split(filedir)
                filename_, _ = os.path.splitext(filename)
                save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
                os.makedirs(save_dir, exist_ok=True)
                feature_rec_np = feat_rec.detach().cpu().numpy()
                np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)

        # Compute prediction (reconstruction error)
        feature_align = torch.sigmoid(feature_align) 
        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W
        pred = self.upsample(pred)  # B x 1 x H x W
        
        # Prepare output dictionary based on available memories
        output_dict = {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
        }
        
        # Add memory-specific outputs if available
        if channel_result is not None:
            output_dict.update({
                "channel_attention": channel_result['att_weight'],
                "channel_scores": channel_result['attention_scores'],
            })
        
        if spatial_result is not None:
            output_dict.update({
                "spatial_attention": spatial_result['att_weight'],
                "spatial_ssim": spatial_result['ssim_similarity'],
            })
        
        return output_dict


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src
        pos = torch.cat(
            [pos.unsqueeze(1)] * src.size(1), dim=1
        )  # (H X W) x B x C

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = tgt
        pos = torch.cat(
            [pos.unsqueeze(1)] * tgt.size(1), dim=1
        )  # (H X W) x B x C

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerMemoryDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Standard transformer decoder layer components
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # Self attention
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        feature_size,
        num_pos_feats=128,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
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
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]), device=tensor.device)  # H x W
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
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size  # H, W
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)  # W
        j = torch.arange(self.feature_size[0], device=tensor.device)  # H
        x_emb = self.col_embed(i)  # W x C // 2
        y_emb = self.row_embed(j)  # H x C // 2
        pos = torch.cat(
            [
                torch.cat(
                    [x_emb.unsqueeze(0)] * self.feature_size[0], dim=0
                ),  # H x W x C // 2
                torch.cat(
                    [y_emb.unsqueeze(1)] * self.feature_size[1], dim=1
                ),  # H x W x C // 2
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (H X W) X C
        return pos

def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    if pos_embed_type in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"):
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed