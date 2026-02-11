# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from wave_dynamic_layer import Dynamic_MLP_OFA, Dynamic_MLP_Decoder
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
import numpy as np

import torch
import torch.nn as nn
import pdb
import math
from functools import reduce
import torch.nn.functional as F
import json
import warnings

from timm.models.vision_transformer import Block

def resize(input,
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

class OFAViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, drop_rate=0.,
                 embed_dim=1024, depth=24, num_heads=16, wv_planes=128, num_classes=45,
                 global_pool=True, mlp_ratio=4., resize=16,norm_layer=nn.LayerNorm):
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        self.patch_embed = Dynamic_MLP_OFA(wv_planes=128, inter_dim=128, kernel_size=16, resize=resize,embed_dim=embed_dim)
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def update_pos_embed(self, num_patches):
        
        cls_pos_embed = self.pos_embed[:, :1, :]
        patch_pos_embed = self.pos_embed[:, 1:, :]
        orig_num_patches = patch_pos_embed.shape[1]
        if num_patches != orig_num_patches:
            
            orig_size = int(math.sqrt(orig_num_patches))
            new_size = int(math.sqrt(num_patches))
            
            patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, -1)
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # [1, embed_dim, orig_size, orig_size]
            patch_pos_embed = F.interpolate(patch_pos_embed, size=(new_size, new_size), mode='bilinear',
                                            align_corners=False)
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
            
            self.pos_embed = nn.Parameter(torch.cat([cls_pos_embed, patch_pos_embed], dim=1))

    def forward_features(self, x, wave_list):
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        num_patches = x.shape[1]
        
        if num_patches != (self.pos_embed.shape[1] - 1):
            self.update_pos_embed(num_patches)

        
        pos_embed = self.pos_embed[:, 1:, :]
        x = x + pos_embed

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward_head(self, x, pre_logits=False):
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, wave_list):
        x = self.forward_features(x, wave_list)
        x = self.forward_head(x)
        return x


def vit_small_patch16(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch32(**kwargs):
    model = OFAViT(
        img_size=448,patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,resize=32,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch8(**kwargs):
    model = OFAViT(
        img_size=224,patch_size=16, embed_dim=768, depth=1, num_heads=12, mlp_ratio=4,resize=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = OFAViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    checkpoint_model = torch.load('D:/code/transfer/DOFA_ViT_base_e120.pth',weights_only=True)
    vit_model = vit_base_patch8(num_classes=10)
    state_dict = vit_model.state_dict()
    
    msg = vit_model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    vit_model = vit_model.cuda()
    # vit_model = vit_base_patch16(num_classes=10).cuda()
    s2_img = torch.randn(10, 9, 448, 448).cuda()
    wavelengths = [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]
    out_feat = vit_model.forward_features(s2_img, wave_list=wavelengths)
    out_logits = vit_model.forward(s2_img, wave_list=wavelengths)
    print(out_feat.shape)
    print(out_logits.shape)