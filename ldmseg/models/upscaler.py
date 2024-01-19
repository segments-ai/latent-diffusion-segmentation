"""
Author: Wouter Van Gansbeke

File with a simple upscaler (decoder) model for latent diffusion training
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
try:
    from diffusers.models.unet_2d_blocks import UNetMidBlock2D
except ImportError:
    print("Diffusers package not found to train VAEs. Please install diffusers")
    raise ImportError


class Upscaler(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        int_channels: int = 256,
        upscaler_channels: int = 256,
        out_channels: int = 128,
        num_mid_blocks: int = 0,
        num_upscalers: int = 1,
        fuse_rgb: bool = False,
        downsample_factor: int = 8,
        norm_num_groups: int = 32,
        pretrained_path: Optional[str] = None,
    ) -> None:

        super().__init__()

        self.enable_mid_block = num_mid_blocks > 0
        self.num_mid_blocks = num_mid_blocks
        self.downsample_factor = downsample_factor
        self.interpolation_factor = self.downsample_factor // (2 ** num_upscalers)

        self.fuse_rgb = fuse_rgb
        multiplier = 2 if self.fuse_rgb else 1
        self.define_decoder(out_channels, int_channels, upscaler_channels, norm_num_groups,
                            latent_channels * multiplier, num_upscalers=num_upscalers)
        self.gradient_checkpoint = False
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
        print('Interpolation factor: ', self.interpolation_factor)

    def enable_gradient_checkpointing(self):
        raise NotImplementedError("Gradient checkpointing not implemented for Upscaler")

    def load_pretrained(self, pretrained_path):
        data = torch.load(pretrained_path, map_location='cpu')
        # remove the module prefix from the state dict
        data['vae'] = {k.replace('module.', ''): v for k, v in data['vae'].items()}
        msg = self.load_state_dict(data['vae'], strict=False)
        print(f'Loaded pretrained decoder from VAE checkp. {pretrained_path} with message {msg}')

    def define_decoder(
        self,
        num_classes: int,
        int_channels: int = 256,
        upscaler_channels: int = 256,
        norm_num_groups: int = 32,
        latent_channels: int = 4,
        num_upscalers: int = 1,
    ):

        decoder_in_conv = nn.Conv2d(latent_channels, int_channels, kernel_size=3, padding=1)

        if self.enable_mid_block:
            decoder_mid_block = [UNetMidBlock2D(
                in_channels=int_channels,
                resnet_eps=1e-6,
                resnet_act_fn='silu',
                output_scale_factor=1,
                resnet_time_scale_shift="default",
                attn_num_head_channels=None,
                resnet_groups=norm_num_groups,
                temb_channels=None,
            ) for _ in range(self.num_mid_blocks)]
        else:
            decoder_mid_block = [nn.Identity()]

        dim = upscaler_channels
        upscaler = []
        for i in range(num_upscalers):
            in_channels = int_channels if i == 0 else dim
            upscaler.extend(
                [
                    nn.ConvTranspose2d(in_channels, dim, kernel_size=2, stride=2),
                    LayerNorm2d(dim),
                    nn.SiLU()
                ]
            )
        upscaler.extend(
            [
                nn.GroupNorm(norm_num_groups, dim),
                nn.SiLU(),
                nn.Conv2d(dim, num_classes, 3, padding=1),
            ]
        )

        self.decoder = nn.Sequential(
            decoder_in_conv,
            *decoder_mid_block,
            *upscaler,
        )

    def freeze_layers(self):
        raise NotImplementedError

    def decode(self, z, interpolate=True):
        x = self.decoder(z)
        if interpolate:
            x = F.interpolate(x, scale_factor=self.interpolation_factor, mode='bilinear', align_corners=False)
        return x

    def forward(
        self,
        z: torch.Tensor,
        interpolate: bool = False,
        z_rgb: Optional[torch.tensor] = None
    ) -> torch.Tensor:

        if z_rgb is not None and self.fuse_rgb:
            z = torch.cat([z, z_rgb], dim=1)

        return self.decode(z, interpolate=interpolate)


class LayerNorm2d(nn.Module):
    # copied from detectron2
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
