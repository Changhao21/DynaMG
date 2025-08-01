# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner import CheckpointLoader
from mmengine.utils import to_2tuple

from mmseg.registry import MODELS
from ..utils.embed import PatchEmbed, PatchMerging
from mmseg.models.decode_heads.fpn_head import FPNHead
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.models.decode_heads.uper_head import UPerHead
class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


@MODELS.register_module()
class SwinTransformer_ML(BaseModule):
    """Swin Transformer backbone.

    This backbone is the implementation of `Swin Transformer:
    Hierarchical Vision Transformer using Shifted
    Windows <https://arxiv.org/abs/2103.14030>`_.
    Inspiration from https://github.com/microsoft/Swin-Transformer.

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int | float): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LN').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None,
                 in_channels_FNP=[96, 192, 384, 768],
                 in_index=[0, 1, 2, 3],
                 feature_strides=[4, 8, 16, 32],
                 channels=256,
                 dropout_ratio=0.1,
                 num_classes=171,
                 norm_cfg_FPN=None,
                 align_corners=False,
                 ):
        self.frozen_stages = frozen_stages
        self.num_classes = num_classes
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super().__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_channels),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        self.loss = CrossEntropyLoss(reduction='mean')
        # self.FPN = FPNHead( in_channels=in_channels_FNP, in_index=in_index,
        # feature_strides=feature_strides,
        # channels=channels,
        # dropout_ratio=dropout_ratio,
        # num_classes=num_classes,
        # norm_cfg=norm_cfg_FPN,
        # align_corners=align_corners,)

        self.decode = UPerHead(
            in_channels=in_channels_FNP,
            in_index=in_index,
            pool_scales=(1, 2, 3, 6),
            channels=channels,
            dropout_ratio=dropout_ratio,
            num_classes=num_classes,
            norm_cfg=norm_cfg_FPN,
            align_corners=align_corners,
        )

        self.mask = None
        self.iter_train = -1

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    print_log('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                if table_key in self.state_dict():
                    table_current = self.state_dict()[table_key]
                    L1, nH1 = table_pretrained.size()
                    L2, nH2 = table_current.size()
                    if nH1 != nH2:
                        print_log(f'Error in loading {table_key}, pass')
                    elif L1 != L2:
                        S1 = int(L1**0.5)
                        S2 = int(L2**0.5)
                        table_pretrained_resized = F.interpolate(
                            table_pretrained.permute(1, 0).reshape(
                                1, nH1, S1, S1),
                            size=(S2, S2),
                            mode='bicubic')
                        state_dict[table_key] = table_pretrained_resized.view(
                            nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, strict=False)

    def forward_backbone(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs

    def resize(self,input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
        if warning:
            if size is not None and align_corners:
                input_h, input_w = tuple(int(x) for x in input.shape[2:])
                output_h, output_w = tuple(int(x) for x in size)
                if output_h > input_h or output_w > output_h:
                    if ((output_h > 1 and output_w > 1 and input_h > 1
                         and input_w > 1) and (output_h - 1) % (input_h - 1)
                            and (output_w - 1) % (input_w - 1)):
                        warnings.warn(
                            f'When align_corners={align_corners}, '
                            'the output would more aligned if '
                            f'input size {(input_h, input_w)} is `x+1` and '
                            f'out size {(output_h, output_w)} is `nx+1`')
        return F.interpolate(input, size, scale_factor, mode, align_corners)

    # def get_mask(self,x,label,flage):
    #     B, C, H, W = x.shape
    #     if flage == 'train':
    #         gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in label]
    #         label = torch.stack(gt_semantic_segs, dim=0)
    #         label[label > self.num_classes - 1] = self.num_classes
    #         mask = torch.zeros(B, self.num_classes + 1, H, W).cuda()
    #         mask = mask.scatter(1, label, 1)
    #         mask = mask[:, :-1, :, :]
    #
    #         x = self.forward_backbone(x)
    #         x = self.FPN(x)
    #         x = self.resize(input=x, size=label.shape[-2:], mode='bilinear', align_corners=False)
    #         label_pre = torch.argmax(x, dim=1, keepdim=True)
    #         mask_pre = torch.zeros(B, self.num_classes, H, W).cuda()
    #         mask_pre = mask_pre.scatter(1, label_pre, 1)
    #
    #         gap = torch.abs(mask_pre - mask).sum(dim=(2, 3))
    #         gap = torch.argmax(gap, dim=1)
    #
    #         mask_reture = []
    #         for i, j in enumerate(gap):
    #             tmp = mask[i, j, :, :]
    #             mask_reture.append(tmp)
    #         mask = torch.stack(mask_reture, dim=0).unsqueeze(1)
    #
    #     else:
    #         mask = torch.ones((B,1,H,W))
    #
    #     return mask.cuda()
    #
    #
    # def forward(self,x,label,flage):
    #     mask = self.get_mask(x,label,flage)
    #     x = x*mask
    #     x = self.forward_backbone(x)
    #     x = self.FPN(x)
    #
    #     return x

    def get_loss_weight_double(self,wieght):
        if wieght<=15000:
            f = 0.1
        else:
            f = 0.0
        return f

    def get_mask_v1(self,x,label,flage):
        B, C, H, W = x.shape

        if flage == 'train':
            gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in label]
            label = torch.stack(gt_semantic_segs, dim=0)
            mask = torch.ones((B, 1, H, W))
            mask[label > self.num_classes - 1] = 0

            x = self.forward_backbone(x)
            x = self.decode(x)
            pre = x
            x = self.resize(input=x, size=label.shape[-2:], mode='bilinear', align_corners=False)
            label_pre = torch.argmax(x, dim=1, keepdim=True)

            gap = torch.abs(label_pre-label)*mask.cuda()

            mask = torch.ones((B, 1, H, W))
            mask[gap < gap.mean()] = 0
            self.iter_train = self.iter_train+1
            weight  = self.get_loss_weight_double(self.iter_train)

        else:
            x = self.forward_backbone(x)
            x = self.decode(x)
            pre = x
            mask = torch.ones((B,1,H,W))
            weight = None

        return mask.cuda(), pre, weight

    def get_mask_v3(self,x,label,flage):
        B, C, H, W = x.shape

        if flage == 'train':
            gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in label]
            label = torch.stack(gt_semantic_segs, dim=0)
            mask = torch.ones((B, 1, H, W))
            mask[label > self.num_classes - 1] = 0

            x = self.forward_backbone(x)
            x = self.decode(x)
            pre = x
            x = self.resize(input=x, size=label.shape[-2:], mode='bilinear', align_corners=False)
            label_pre = torch.argmax(x, dim=1, keepdim=True)

            gap = torch.abs(label_pre-label)*mask.cuda()

            mask = torch.zeros((B, 1, H, W))
            mask[gap < gap.mean()] = 1
            self.iter_train = self.iter_train+1
            weight  = 0.5

            tmp_thr = mask.sum() / (B * H * W)

        else:
            x = self.forward_backbone(x)
            x = self.decode(x)
            pre = x
            mask = torch.ones((B,1,H,W))
            weight = None
            tmp_thr = None

        return mask.cuda(), pre, weight, tmp_thr








    def get_mask_v2(self,x,label,flage):
        B, C, H, W = x.shape

        if flage == 'train':
            gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in label]
            label = torch.stack(gt_semantic_segs, dim=0)
            mask = torch.ones((B, 1, H, W))
            mask[label > self.num_classes - 1] = 0

            x = self.forward_backbone(x)
            x = self.decode(x)
            pre = x
            x = self.resize(input=x, size=label.shape[-2:], mode='bilinear', align_corners=False)
            label_pre = torch.argmax(x, dim=1, keepdim=True)

            gap = torch.abs(label_pre-label)*mask.cuda()

            mask = torch.ones((B, 1, H, W))
            mask[gap < gap.mean()] = 0

            thr = 0.5
            tmp_thr = mask.sum()/(B*H*W)
            if tmp_thr < thr:
                mask = torch.ones((B, 1, H, W))
                weight = 1.0
            else:
                weight = 0.1

        else:
            x = self.forward_backbone(x)
            x = self.decode(x)
            pre = x
            mask = torch.ones((B,1,H,W))
            weight = None
            tmp_thr = None

        return mask.cuda(), pre, weight, tmp_thr

    def get_mask_ab_womsk(self,x,label,flage):
        B, C, H, W = x.shape

        x = self.forward_backbone(x)
        x = self.decode(x)
        pre = x
        mask = torch.ones((B,1,H,W))
        weight = 1.0

        return mask.cuda(), pre, weight


    def get_mask_ab_1_mask(self,x,label,flage):
        B, C, H, W = x.shape

        if flage == 'train':
            gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in label]
            label = torch.stack(gt_semantic_segs, dim=0)
            mask = torch.ones((B, 1, H, W))
            mask[label > self.num_classes - 1] = 0

            x = self.forward_backbone(x)
            x = self.decode(x)
            pre = x
            x = self.resize(input=x, size=label.shape[-2:], mode='bilinear', align_corners=False)
            label_pre = torch.argmax(x, dim=1, keepdim=True)

            gap = torch.abs(label_pre-label)*mask.cuda()

            mask = torch.ones((B, 1, H, W))
            mask[gap < gap.mean()] = 0
            self.iter_train = self.iter_train+1

            tmp_thr = mask.sum() / (B * H * W)

        else:
            x = self.forward_backbone(x)
            x = self.decode(x)
            pre = x
            mask = torch.ones((B,1,H,W))

            tmp_thr = None

        return mask.cuda(), pre, tmp_thr


    def get_mask_random(self,x,label,flage):
        B, C, H, W = x.shape

        if flage == 'train':

            x = self.forward_backbone(x)
            x = self.decode(x)
            pre = x

            mask = torch.zeros((B, 1, H, W))

            pos = torch.rand_like(mask) > 0.5

            mask[pos] = 1

            tmp_thr = mask.sum() / (B * H * W)

        else:
            x = self.forward_backbone(x)
            x = self.decode(x)
            pre = x
            mask = torch.ones((B,1,H,W))
            weight = None
            tmp_thr = None

        return mask.cuda(), pre,  tmp_thr

    def get_loss_weight_linear(self,wieght):
        total_val = 40000
        return 1.0-(wieght/total_val)


    def get_loss_weight_exp(self,wieght):
        a, b = 1e-17, 4.0
        f = 1.0*math.exp(-a * wieght ** b)
        return f

    def get_mask_v3_linear(self,x,label,flage):
        B, C, H, W = x.shape

        if flage == 'train':
            gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in label]
            label = torch.stack(gt_semantic_segs, dim=0)
            mask = torch.ones((B, 1, H, W))
            mask[label > self.num_classes - 1] = 0

            x = self.forward_backbone(x)
            x = self.decode(x)
            pre = x
            x = self.resize(input=x, size=label.shape[-2:], mode='bilinear', align_corners=False)
            label_pre = torch.argmax(x, dim=1, keepdim=True)

            gap = torch.abs(label_pre-label)*mask.cuda()

            mask = torch.zeros((B, 1, H, W))
            mask[gap < gap.mean()] = 1

            self.iter_train = self.iter_train+1
            weight = self.get_loss_weight_exp(self.iter_train)

            tmp_thr = mask.sum() / (B * H * W)

        else:
            x = self.forward_backbone(x)
            x = self.decode(x)
            pre = x
            mask = torch.ones((B,1,H,W))
            weight = None
            tmp_thr = None

        return mask.cuda(), pre, weight



    # def get_loss_weight_linear(self,wieght):
    #     total_val = 40000
    #     return 1.0-(wieght/total_val)

    # def get_loss_weight_exp(self,wieght):
    #     a, b = 1e-17, 4.0
    #     f = 1.0*math.exp(-a * wieght ** b)
    #     return f

    # def get_loss_weight_double(self,wieght):
    #     if wieght<=20000:
    #         f = 0.1
    #     else:
    #         f = 0.0
    #     return f


    # def get_loss_weight_exp(self,wieght):
    #     a, b = 1e-19, 4
    #     f = 0.1*math.exp(-a * wieght ** b)
    #     return f


    #final==========================================================================
    # def forward(self,x,label,flage):
    #     mask, pre, weight, tmp_thr = self.get_mask_v3(x,label,flage)
    #     x = x*mask
    #     x = self.forward_backbone(x)
    #     x = self.decode(x)
    #     pre_mask = x
    #
    #     return [pre, pre_mask, weight, tmp_thr]
    #final============================================================================

    # def forward(self,x,label,flage):
    #     # mask, pre, weight = self.get_mask_ab_1_mask(x,label,flage)
    #     # mask, pre, weight = self.get_mask_ab_womsk(x, label, flage)
    #     # mask, pre, weight = self.get_mask_random(x, label, flage)
    #     mask, pre, weight = self.get_mask_v3_linear(x, label, flage)
    #     x = x*mask
    #     x = self.forward_backbone(x)
    #     x = self.decode(x)
    #     pre_mask = x
    #
    #     return [pre, pre_mask, weight]
        # return pre

    def forward(self,x):
        x = self.forward_backbone(x)
        x = self.decode(x)

        return x





