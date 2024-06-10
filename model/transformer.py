# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Transformer(nn.Module):

    def __init__(self, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 hidden_dim=256, sr_ratios=[8, 4, 2, 1], num_encoder_layers=[1, 1, 1, 1],
                 num_decoder_layers=[1, 1, 1, 1], grid_sizes=[8, 4, 2, 1], query_size=()):
        super().__init__()
        self.encoder = TransformerEncoder(embed_dims, num_heads, mlp_ratios, num_encoder_layers, norm_layer,
                                          sr_ratios=sr_ratios, out_dim=hidden_dim)
        self.decoder = TransformerDecoder(hidden_dim, num_heads[::-1], mlp_ratios[::-1], num_decoder_layers[::-1],
                                          norm_layer, grid_sizes=grid_sizes[::-1], sr_ratio=sr_ratios[0],
                                          query_size=query_size)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, query_embed, pos_embed=None, mask=None):
        # flatten NxCxHxW to HWxNxC
        B, _, H, W = features[0].shape
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(0).repeat(B, 1, 1)

        # tgt = torch.zeros_like(query_embed)
        memory = self.encoder(features, pos_embed)
        hs = self.decoder(memory, query_embed)
        return hs.permute(0, 2, 1).reshape(B, -1, H, W)


class TransformerEncoder(nn.Module):

    def __init__(self, embed_dims, num_heads, mlp_ratios, num_encoder_layers, norm_layer, sr_ratios, out_dim=256):
        super().__init__()
        # self.layers = _get_clones(encoder_layer, num_layers)
        self.layers = nn.ModuleList()
        self.linear = nn.ModuleList()
        for i, num in enumerate(num_encoder_layers):
            #layer = [EncoderLayer(embed_dims[i], num_heads[i], mlp_ratios[i],
            #                      norm_layer=norm_layer, sr_ratio=sr_ratios[i]) for _ in range(num)]
            layer = EncoderLayer(embed_dims[i], num_heads[i], mlp_ratios[i],
                                  norm_layer=norm_layer, sr_ratio=sr_ratios[i])
            self.layers.append(layer)
            #self.layers.append(nn.Sequential(*layer))
            self.linear.append(nn.Linear(embed_dims[i], out_dim))
        self.embed_dims = embed_dims

    def forward(self, features, pos=None, mask=None):
        if pos:
            features = [feature_ + pos_ for feature_, pos_ in zip(features, pos)]
        outputs = []
        for feature, layer, linear in zip(features, self.layers, self.linear):
            B, C, H, W = feature.shape
            feature = feature.reshape(B, C, -1).permute(0, 2, 1)
            feature_size = [H, W]
            output = layer(feature, feature_size)
            output = linear(output).permute(0, 2, 1).reshape(B, -1, H, W)
            outputs.append(output)
        return outputs


class TransformerDecoder(nn.Module):

    def __init__(self, embed_dims, num_heads, mlp_ratios, num_decoder_layers,
                 norm_layer, grid_sizes, sr_ratio, query_size=(120, 160)):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, num in enumerate(num_decoder_layers):
            '''
            layer = [DecoderLayer(embed_dims[i], num_heads[i], mlp_ratios[i],
                                  norm_layer=norm_layer, grid_size=grid_sizes[i],
                                  sr_ratio=sr_ratio)
                     for _ in range(num)]
            self.layers.append(nn.Sequential(*layer))'''
            layer = DecoderLayer(embed_dims, num_heads[i], mlp_ratios[i],
                                 norm_layer=norm_layer, grid_size=grid_sizes[i], query_size=query_size)
            self.layers.append(layer)
        self.embed_dims = embed_dims

    def forward(self, features, query_pos=None, pos=None):
        output = query_pos
        features = features[::-1]

        for feature, layer in zip(features, self.layers):
            B, C, H, W = feature.shape
            feature = feature.reshape(B, C, -1).permute(0, 2, 1)
            output = layer(output, feature, feature_size=(H, W))

        return output


class EncoderLayer(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, feature_size):
        x_ = self.norm1(x)
        x = x + self.drop_path(self.attn(q=x_, kv=x_, feature_size=feature_size))
        x = x + self.drop_path(self.mlp(self.norm2(x), feature_size))

        return x


class DecoderLayer(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, grid_size=1, query_size=(120, 160)):
        super().__init__()
        '''
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)'''
        self.attn = ShortRangeAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, grid_size=grid_size, query_size=query_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.query_size = query_size

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, q, kv, feature_size):
        #q = q + self.drop_path(self.self_attn(q=q, kv=q, feature_size=[120, 160]))
        #q = self.norm1(q)
        q = q + self.drop_path(self.attn(q=q, kv=kv, feature_size=feature_size))
        q = self.norm2(q)
        q = q + self.drop_path(self.mlp(q, feature_size=self.query_size))
        q = self.norm3(q)

        return q


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, feature_size):
        H, W = feature_size
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, q, kv, feature_size):
        B, N, C = q.shape
        H, W = feature_size
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            kv_ = kv.permute(0, 2, 1).reshape(B, C, H, W)
            kv_ = self.sr(kv_).reshape(B, C, -1).permute(0, 2, 1)
            kv_ = self.norm(kv_)
            kv = self.kv(kv_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = kv.reshape(B, C, -1).permute(0, 2, 1)
            kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ShortRangeAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., grid_size=1, query_size=[120, 160]):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.H = query_size[0]
        self.W = query_size[1]

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.grid_size = grid_size

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def short_range_grid_partition(self, features, grid_size):
        """Partition the input feature maps into non-overlapping windows.
        Args:
          features: [B, C, H, W] feature maps.
        Returns:
          Partitioned features: [B, nH, nW, c, wSize, wSize].
        Raises:
          ValueError: If the feature map sizes are not divisible by window sizes.
        """

        _, c, h, w = features.shape
        # grid_size = config.grid_size
        if h % grid_size != 0 or w % grid_size != 0:
            raise ValueError(f'Feature map sizes {(h, w)} '
                             f'not divisible by window size ({grid_size}).')
        features = features.reshape(-1, c, grid_size, h // grid_size, grid_size, w // grid_size)
        features = features.permute(0, 2, 4, 1, 3, 5)
        features = features.reshape(-1, c, h // grid_size, w // grid_size)
        return features

    def short_range_grid_stitch_back(self, features, grid_rate, h, w):
        """Reverse window_partition."""
        c = features.shape[1]
        features = features.reshape(-1, grid_rate, grid_rate, c, h // grid_rate, w // grid_rate)
        return features.permute(0, 3, 1, 4, 2, 5).reshape(-1, c, h, w)

    def forward(self, q, kv, feature_size):
        B, N, C = q.shape
        H, W = feature_size
        # q = q.reshape(B, C, -1).permute(0, 2, 1)
        q = self.q(q)
        kv = self.kv(kv)

        if self.grid_size > 1:
            q_ = q.permute(0, 2, 1).reshape(B, C, self.H, self.W)
            q_ = self.short_range_grid_partition(q_, self.grid_size)
            b, _, h, w = q_.shape
            q = q_.reshape(b, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)

            kv_ = kv.permute(0, 2, 1).reshape(B, -1, H, W)
            kv_ = self.short_range_grid_partition(kv_, self.grid_size)
            kv = kv_.reshape(b, 2, self.num_heads, C // self.num_heads, -1).permute(1, 0, 2, 4, 3)
        else:
            q = q.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        #print('grid:', self.grid_size, 'q k v:', q.shape, k.shape, v.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        if self.grid_size > 1:
            x = x.transpose(2, 3).reshape(b, C, -1)  # b, C, n
            x = self.short_range_grid_stitch_back(x, self.grid_size, feature_size[0], feature_size[1])
            x = x.reshape(B, C, N).permute(0, 2, 1)
        else:
            x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
