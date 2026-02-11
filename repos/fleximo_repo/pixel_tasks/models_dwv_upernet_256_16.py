import torch
import numpy as np
import math
import itertools
import torch.nn.functional as F
from torch import distributed as dist
from torch import nn, einsum
from einops import rearrange
from timm.models.vision_transformer import DropPath, Mlp
from functools import partial

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
from .wave_dynamic_layer_seg import Dynamic_MLP_OFA, Dynamic_MLP_Decoder
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


class OFAViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=256, patch_size=16, drop_rate=0.,
                 embed_dim=1024, depth=24, num_heads=16, wv_planes=128, num_classes=2,
                 global_pool=True, mlp_ratio=4., resize=16, norm_layer=nn.LayerNorm):
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        self.patch_embed = Dynamic_MLP_OFA(wv_planes=128, inter_dim=128, kernel_size=16, resize=resize,
                                           embed_dim=embed_dim)
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        # self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

        self.cls_seg = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=3, padding=1),
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(768, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, 16, 16),  # 2048, 16, 16
            nn.Dropout(0.5)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(768, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 512, 8, 8),  # 2048, 16, 16
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(768, 1024, 1, 1),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024, 1024, 4, 4),  # 2048, 16, 16
            nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(768, 2048, 1, 1),
            nn.GroupNorm(32, 2048),
            nn.GELU(),
            nn.ConvTranspose2d(2048, 2048, 2, 2),  # 2048, 16, 16
            nn.Dropout(0.5)
            # 2048, 16, 16
        )

        self.decoder = FPNHEAD()

        torch.nn.init.normal_(self.head.weight, std=0.02)


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
            print("Resize the pos_embedding successfully !!!")

    def forward(self, x, wave_list):
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
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        seg1 = x

        B = x.shape[0]
        # print(x.shape)

        seg1 = seg1.reshape(B, 16, 16, 768).permute(0, 3, 1, 2).contiguous()

        m = {}
        m[0] = self.conv0(seg1)  # 256,128,128
        # print(m[0].shape)

        m[1] = self.conv1(seg1)  # 512,64,64
        # print(m[1].shape)

        m[2] = self.conv2(seg1)  # 1024,32,32
        # print(m[2].shape)

        m[3] = self.conv3(seg1)  # 2048,16,16
        # print(m[3].shape)

        m = list(m.values())
        x = self.decoder(m)
        x = self.cls_seg(x)


        return {'out': x}
        # return x


class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear',
                                                align_corners=True)
            out_puts.append(ppm_out)
        return out_puts

class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=13):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=1),
            # nn.BatchNorm2d(self.out_channels),
            nn.GroupNorm(16, self.out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out

class FPNHEAD(nn.Module):
    def __init__(self, channels=2048, out_channels=256):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels // 2, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels // 4, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels // 8, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, input_fpn):
        # b, 512, 7, 7
        x1 = self.PPMHead(input_fpn[-1])

        x = nn.functional.interpolate(x1, size=(x1.size(2) * 2, x1.size(3) * 2), mode='bilinear', align_corners=True)
        x = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2])
        x2 = self.Conv_fuse1_(x)

        x = nn.functional.interpolate(x2, size=(x2.size(2) * 2, x2.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse2(input_fpn[-3])
        x3 = self.Conv_fuse2_(x)

        x = nn.functional.interpolate(x3, size=(x3.size(2) * 2, x3.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse3(input_fpn[-4])
        x4 = self.Conv_fuse3_(x)

        x1 = F.interpolate(x1, x4.size()[-2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-2:], mode='bilinear', align_corners=True)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))

        return x







def vit_base_patch16_64(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, resize=64,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16_32(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, resize=32,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16_16(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, resize=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16_8(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, resize=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16_4(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, resize=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def vit_base_patch16(**kwargs):
    model = OFAViT(
        img_size=448,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch32(**kwargs):
    model = OFAViT(
        img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, resize=32,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    checkpoint_model = torch.load('/media/ps/sda2/LXY/croma_linux2/DOFA/pth/DOFA_ViT_base_e100.pth', weights_only=True)
    vit_model = vit_base_patch16_16(num_classes=10)
    state_dict = vit_model.state_dict()
    msg = vit_model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    vit_model = vit_model.cuda()
    # vit_model = vit_base_patch16(num_classes=10).cuda()
    s2_img = torch.randn(2, 3, 256, 256).cuda()
    wavelengths = [0.665, 0.56, 0.49]
    out_logits = vit_model.forward(s2_img, wave_list=wavelengths)
    print(out_logits.shape)
