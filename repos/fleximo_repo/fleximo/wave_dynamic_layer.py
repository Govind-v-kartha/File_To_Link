import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# from flex_patch_embed import FlexiPatchEmbed
import collections.abc
from itertools import repeat
from typing import Any, Optional, Tuple, Union
from torch import Tensor
from einops import rearrange
# from functorch import vmap
from torch import vmap


def to_2tuple(x: Any) -> Tuple:
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


def pi_resize_patch_embed(
        patch_embed: Tensor,
        new_patch_size: Tuple[int, int],
        interpolation: str = "bicubic",
        antialias: bool = True,
):
    """Resample patch embedding weights to a target resolution via pseudo-inverse
    resizing.

    Based on:
        https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py
        https://arxiv.org/abs/2212.08013

    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [d, c h', w']
    """
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4D tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (height, width)"

    old_patch_size = tuple(patch_embed.shape[2:])

    # Return original kernel if no resize is necessary
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x: Tensor, shape: Tuple[int, int]):
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=interpolation,
            antialias=antialias,
        )
        return x_resized[0, 0, ...]

    def calculate_pinv(old_shape: Tuple[int, int], new_shape: Tuple[int, int]):
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    # Calculate pseudo-inverse of resize matrix
    resize_matrix_pinv = calculate_pinv(old_patch_size, new_patch_size)
    resize_matrix_pinv = resize_matrix_pinv.to(patch_embed.device)

    def resample_patch_embed(patch_embed: Tensor):
        h, w = new_patch_size
        resampled_kernel = resize_matrix_pinv @ patch_embed.reshape(-1)
        return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

    v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

    return v_resample_patch_embed(patch_embed)


def interpolate_resize_patch_embed(
        patch_embed: Tensor,
        new_patch_size: Tuple[int, int],
        interpolation: str = "bicubic",
        antialias: bool = True,
):
    """Resample patch embedding weights to a target resolution via interpolation

    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [d, c h', w']
    """
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4D tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (height, width)"

    patch_embed = F.interpolate(
        patch_embed, new_patch_size, mode=interpolation, antialias=antialias
    )

    return patch_embed


random_seed = 1234
torch.manual_seed(random_seed)


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num: -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[-1]
        )  # Using the last output to generate bias
        return weights, bias


class Basic1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(
            conv,
        )
        if not bias:
            self.conv.add_module("ln", nn.LayerNorm(out_channels))
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class Dynamic_MLP_Decoder(nn.Module):
    def __init__(self, wv_planes, inter_dim=128, kernel_size=16, decoder_embed=512):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.inter_dim = inter_dim
        self.decoder_embed = decoder_embed
        self._num_kernel = self.kernel_size * self.kernel_size * self.decoder_embed

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, decoder_embed
        )
        self.scaler = 0.01

        self._init_weights()

    def _get_weights(self, waves, batch=True):
        dweights = []
        dynamic_weights = None
        if batch:
            dynamic_weights = self.weight_generator(waves)
        else:
            for i in range(waves.size(0)):
                dweights.append(self.weight_generator(waves[i]))
            dynamic_weights = torch.stack(dweights, dim=0)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)

    def forward(self, img_feat, waves):
        inplanes = waves.size(0)
        # wv_feats: 9,128 -> 9*16*16,512
        weight, bias = self._get_weights(waves)  # 9,16*16*512
        dynamic_weight = weight.view(
            inplanes * self.kernel_size * self.kernel_size, self.decoder_embed
        )  # 9*16*16,512
        weights = dynamic_weight * self.scaler

        dynamic_out = F.linear(img_feat, weights, bias=None)
        x = dynamic_out
        return x


class Dynamic_MLP_OFA(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024, resize=32):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01
        self.resize = resize

        self.patch_size_seq = (8, 10, 12, 15, 16, 20, 24, 30, 40, 48),

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()


    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)


    def forward(self, img_feat, wvs):

        inplanes = wvs.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3

        # Fix bug        
        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)  # 3,16,16,1024

        if self.kernel_size == self.resize:
            # original
            dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])  # 1024,3,16,16
            # print(dynamic_weight.shape)
        else:
            # Resize
            dynamic_weight = dynamic_weight.permute([0, 3, 1, 2])  # 3,1024,16,16
            # print(dynamic_weight.shape)
            dynamic_weight = pi_resize_patch_embed(dynamic_weight, (self.resize, self.resize))  # 3,1024,8,8
            # print(dynamic_weight.shape)
            dynamic_weight = dynamic_weight.permute([1, 0, 2, 3])  # 3,1024,16,16
            # print(dynamic_weight.shape)

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        # dynamic_out = F.conv2d(
        #     img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        # )
        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.resize, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


if __name__ == '__main__':
    # wavelengths = [3.75, 3.75]
    # sar = torch.rand(1, 2, 224, 224).cuda()  # output:1,196,1024

    # wavelengths = [3.75, 3.75, 3.75, 3.75, 3.75]
    # sar = torch.rand(1, 5, 224, 224).cuda()  # output:1,196,1024

    wavelengths = [3.75, 3.75, 3.75]
    sar = torch.rand(1, 3, 224, 224).cuda()  # output:1,144,1024

    wavelist = torch.tensor(wavelengths, device=sar.device).float()
    patch_embed = Dynamic_MLP_OFA(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=1024, resize=32).cuda()
    x, _ = patch_embed(sar, wavelist)
    print(x.shape)
