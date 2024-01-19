"""
Author: Wouter Van Gansbeke

File with VAE models for latent diffusion training
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional, Union
from ldmseg.utils import OutputDict
try:
    from diffusers.models.unet_2d_blocks import UNetMidBlock2D
    from diffusers import AutoencoderKL
except ImportError:
    print("Diffusers package not found to train VAEs. Please install diffusers")
    raise ImportError


class RangeDict(OutputDict):
    min: torch.Tensor
    max: torch.Tensor


class VAEOutput(OutputDict):
    sample: torch.Tensor
    posterior: torch.Tensor


class EncoderOutput(OutputDict):
    latent_dist: torch.Tensor


class GeneralVAEImage(AutoencoderKL):

    def set_scaling_factor(self, scaling_factor):
        self.scaling_factor = scaling_factor


class GeneralVAESeg(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        int_channels: int = 256,
        out_channels: int = 128,
        block_out_channels: Tuple[int] = (32, 64, 128, 256),
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        scaling_factor: float = 0.18215,
        pretrained_path: Optional[str] = None,
        encoder: Optional[nn.Module] = None,
        num_mid_blocks: int = 0,
        num_latents: int = 2,
        num_upscalers: int = 1,
        upscale_channels: int = 256,
        parametrization: str = 'gaussian',
        fuse_rgb: bool = False,
        resize_input: bool = False,
        act_fn: str = 'none',
        clamp_output: bool = False,
        freeze_codebook: bool = False,
        skip_encoder: bool = False,
    ) -> None:

        super().__init__()

        self.enable_mid_block = num_mid_blocks > 0
        self.num_mid_blocks = num_mid_blocks
        self.downsample_factor = 2 ** (len(block_out_channels) - 1)
        self.interpolation_factor = self.downsample_factor // (2 ** num_upscalers)
        if "discrete" in parametrization:
            num_embeddings = 128
            if freeze_codebook:
                print('Freezing codebook')
                gen = torch.Generator().manual_seed(42)
                Q = torch.linalg.qr(torch.randn(num_embeddings, latent_channels, generator=gen))[0]
                self.codebook = nn.Embedding.from_pretrained(Q, freeze=True)
            else:
                self.codebook = nn.Embedding(num_embeddings, latent_channels, max_norm=None)
            num_latents = num_embeddings // latent_channels
        elif parametrization == 'auto':
            num_latents = 1

        if encoder is None:
            if fuse_rgb:
                in_channels += 3
            self.define_encoder(in_channels, block_out_channels, int_channels, norm_num_groups,
                                latent_channels, num_latents=num_latents, resize_input=resize_input,
                                skip_encoder=skip_encoder)
        else:
            self.encoder = encoder
            self.freeze_encoder()

        self.define_decoder(out_channels, int_channels, norm_num_groups, latent_channels, 
                            num_upscalers=num_upscalers, upscale_channels=upscale_channels)
        self.scaling_factor = scaling_factor
        self.gradient_checkpoint = False
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
        self.parametrization = parametrization
        self.interpolation_factor = self.downsample_factor // (2 ** num_upscalers)
        self.num_latents = num_latents
        self.act_fn = act_fn
        self.clamp_output = clamp_output
        print('Interpolation factor: ', self.interpolation_factor)
        print('Parametrization: ', self.parametrization)
        print('Activation function: ', self.act_fn)
        assert self.parametrization in ['gaussian', 'discrete_gumbel_softmax', 'discrete_codebook', 'auto']
        assert self.num_latents in [1, 2, 32]

    def enable_gradient_checkpointing(self):
        raise NotImplementedError("Gradient checkpointing not implemented for a shallow VAE")

    def load_pretrained(self, pretrained_path):
        data = torch.load(pretrained_path, map_location='cpu')
        # remove the module prefix from the state dict
        data['vae'] = {k.replace('module.', ''): v for k, v in data['vae'].items()}
        msg = self.load_state_dict(data['vae'], strict=True)
        print(f'Loaded pretrained VAE from {pretrained_path} with message {msg}')

    def define_decoder(
        self,
        num_classes: int,
        int_channels: int = 256,
        norm_num_groups: int = 32,
        latent_channels: int = 4,
        num_upscalers: int = 1,
        upscale_channels: int = 256,
    ):

        decoder_in_conv = nn.Conv2d(latent_channels, int_channels, kernel_size=3, padding=1)

        if self.enable_mid_block:
            decoder_mid_block = UNetMidBlock2D(
                in_channels=int_channels,
                resnet_eps=1e-6,
                resnet_act_fn='silu',
                output_scale_factor=1,
                resnet_time_scale_shift="default",
                resnet_groups=norm_num_groups,
                temb_channels=None,
                add_attention=False,
            )
        else:
            decoder_mid_block = nn.Identity()

        dim = upscale_channels
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
            decoder_mid_block,
            *upscaler,
        )

    def define_encoder(
        self,
        in_channels: int,
        block_out_channels: Tuple[int],
        int_channels: int = 256,
        norm_num_groups: int = 32,
        latent_channels: int = 4,
        num_latents: int = 2,
        resize_input: bool = False,
        skip_encoder: bool = False,
    ):
        # define semseg encoder
        if skip_encoder:
            self.encoder = nn.Conv2d(in_channels, latent_channels * num_latents, 8, stride=8)
            return

        encoder_in_block = [
            nn.Conv2d(in_channels, block_out_channels[0] if not resize_input else int_channels,
                      kernel_size=3, padding=1),
            nn.SiLU(),
        ]

        if not resize_input:
            down_blocks_semseg = []
            for i in range(len(block_out_channels) - 1):
                channel_in = block_out_channels[i]
                channel_out = block_out_channels[i + 1]
                down_blocks_semseg.extend(
                    [
                        nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1),
                        nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2),
                        nn.SiLU(),
                    ]
                )
        else:
            down_blocks_semseg = [
                nn.Upsample(scale_factor=1. / self.downsample_factor, mode='bilinear', align_corners=False)
            ]
        encoder_down_blocks = [
            *down_blocks_semseg,
            nn.Conv2d(block_out_channels[-1], int_channels, kernel_size=3, padding=1),
        ]

        encoder_mid_blocks = []
        if self.enable_mid_block:
            for _ in range(self.num_mid_blocks):
                encoder_mid_blocks.append(UNetMidBlock2D(
                    in_channels=int_channels,
                    resnet_eps=1e-6,
                    resnet_act_fn='silu',
                    output_scale_factor=1,
                    resnet_time_scale_shift="default",
                    resnet_groups=norm_num_groups,
                    temb_channels=None,
                    add_attention=False,
                ))
        else:
            encoder_mid_blocks = [nn.Identity()]

        encoder_out_block = [
            nn.GroupNorm(num_channels=int_channels, num_groups=norm_num_groups, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(int_channels, latent_channels * num_latents, 3, padding=1)
        ]

        self.encoder = nn.Sequential(
            *encoder_in_block,
            *encoder_down_blocks,
            *encoder_mid_blocks,
            *encoder_out_block,
        )

    def freeze_layers(self):
        raise NotImplementedError

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)

    def encode(self, semseg):
        moments = self.encoder(semseg)
        if self.parametrization == 'gaussian':
            posterior = DiagonalGaussianDistribution(
                moments, clamp_output=self.clamp_output, act_fn=self.act_fn)
        elif self.parametrization == 'discrete_gumbel_softmax':
            posterior = GumbelSoftmaxDistribution(
                moments, self.codebook, clamp_output=self.clamp_output, act_fn=self.act_fn)
        elif self.parametrization == 'discrete_codebook':
            posterior = DiscreteCodebookAssignemnt(
                moments, self.codebook, clamp_output=self.clamp_output, act_fn=self.act_fn)
        elif self.parametrization == 'auto':
            posterior = Bottleneck(moments, act_fn=self.act_fn)
        return EncoderOutput(latent_dist=posterior)

    def decode(self, z, interpolate=True):
        x = self.decoder(z)
        if interpolate:
            x = F.interpolate(x, scale_factor=self.interpolation_factor, mode='bilinear', align_corners=False)
        return x

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        rgb_sample: Optional[torch.FloatTensor] = None,
        valid_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[VAEOutput, torch.FloatTensor]:

        x = sample

        # encode
        if rgb_sample is not None:
            x = torch.cat([x, rgb_sample], dim=1)

        posterior = self.encode(x).latent_dist

        # sample from posterior
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        # (optional) mask out invalid pixels
        if valid_mask is not None:
            z = z * valid_mask[:, None]

        # decode
        dec = self.decode(z, interpolate=False)

        if not return_dict:
            return (dec,)
        return VAEOutput(sample=dec, posterior=posterior)


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


class Bottleneck(object):
    """
    Simple bottleneck class to be used in the AE.
    """

    def __init__(
        self,
        parameters: torch.Tensor,
        act_fn: str = 'none',
    ):
        self.mean = parameters
        self.mean = self.to_range(self.mean, act_fn)
        self.act_fn = act_fn

    def to_range(self, x, act_fn):
        if act_fn == 'sigmoid':
            return 2 * F.sigmoid(x) - 1
        elif act_fn == 'tanh':
            return F.tanh(x)
        elif act_fn == 'clip':
            return torch.clamp(x, -5.0, 5.0)
        elif act_fn == 'l2':
            return F.normalize(x, dim=1, p=2)
        elif act_fn == 'none':
            return x
        else:
            raise NotImplementedError

    def mode(self):
        return self.mean

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        return self.mean

    def kl(self):
        return torch.sum(torch.pow(self.mean, 2), dim=[1, 2, 3])

    def get_range(self):
        return RangeDict(min=self.mean.min(), max=self.mean.max())

    def __str__(self) -> str:
        return f"Bottleneck(mean={self.mean}, " \
               f"act_fn={self.act_fn})"


class DiagonalGaussianDistribution(object):
    """
    Parametrizes a diagonal Gaussian distribution with a mean and log-variance.
    Allows computing the KL divergence with a standard diagonal Gaussian distribution.
    Added functionalities to diffusers library: bottleneck clamp and activation function.
    """

    def __init__(
        self,
        parameters: torch.Tensor,
        clamp_output: bool = False,
        act_fn: str = 'none',
    ):

        self.parameters = parameters
        if clamp_output:
            parameters = torch.clamp(parameters, -5.0, 5.0)
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.mean = self.to_range(self.mean, act_fn)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.clamp_output = clamp_output
        self.act_fn = act_fn

    def to_range(self, x, act_fn):
        if act_fn == 'sigmoid':
            return 2 * F.sigmoid(x) - 1
        elif act_fn == 'tanh':
            return F.tanh(x)
        elif act_fn == 'clip':
            return torch.clamp(x, -1, 1)
        elif act_fn == 'none':
            return x
        else:
            raise NotImplementedError

    def mode(self):
        return self.mean

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        sample = torch.randn(
            self.mean.shape, generator=generator, device=self.parameters.device, dtype=self.parameters.dtype)
        x = self.mean + self.std * sample
        return x

    def kl(self):
        return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])

    def get_range(self):
        return RangeDict(min=self.mean.min(), max=self.mean.max())

    def __str__(self) -> str:
        return f"DiagonalGaussianDistribution(mean={self.mean}, var={self.var}, " \
               f"clamp_output={self.clamp_output}, act_fn={self.act_fn})"


class GumbelSoftmaxDistribution(object):
    """
    Parametrizes a uniform gumbel softmax distribution.
    """

    def __init__(
        self,
        parameters: torch.Tensor,
        codebook: nn.Embedding,
        clamp_output: bool = False,
        act_fn: str = 'none',
    ):

        self.parameters = parameters
        if clamp_output:
            parameters = torch.clamp(parameters, -5.0, 5.0)

        self.clamp_output = clamp_output
        self.act_fn = act_fn
        self.straight_through = True
        self.temp = 0.2
        self.codebook = codebook
        self.num_tokens = codebook.weight.shape[0]
        assert self.num_tokens == 128
        assert self.parameters.shape[1] == self.num_tokens

    def to_range(self, x, act_fn):
        if act_fn == 'sigmoid':
            return 2 * F.sigmoid(x) - 1
        elif act_fn == 'tanh':
            return F.tanh(x)
        elif act_fn == 'clip':
            return torch.clamp(x, -1, 1)
        elif act_fn == 'none':
            return x
        else:
            raise NotImplementedError

    def mode(self) -> torch.FloatTensor:
        indices = self.get_codebook_indices()
        # one_hot = torch.scatter(torch.zeros_like(self.parameters), 1, indices[:, None], 1.0)
        one_hot = F.one_hot(indices, num_classes=self.num_tokens).permute(0, 3, 1, 2).float()
        sampled = torch.einsum('b n h w, n d -> b d h w', one_hot, self.codebook.weight)
        return sampled

    def get_codebook_indices(self) -> torch.LongTensor:
        return self.parameters.argmax(dim=1)

    def get_codebook_probs(self) -> torch.FloatTensor:
        return nn.Softmax(dim=1)(self.parameters)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        soft_one_hot = F.gumbel_softmax(self.parameters, tau=self.temp, dim=1, hard=self.straight_through)
        sampled = torch.einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        return sampled

    def kl(self) -> torch.FloatTensor:
        logits = rearrange(self.parameters, 'b n h w -> b (h w) n')
        qy = F.softmax(logits, dim=-1)
        log_qy = torch.log(qy + 1e-10)
        log_uniform = torch.log(torch.tensor([1. / self.num_tokens], device=log_qy.device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)
        return kl_div

    def get_range(self):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"DiscreteGumbelSoftmaxDistribution(mean={self.parameters}, " \
               f"clamp_output={self.clamp_output}, act_fn={self.act_fn})"


class DiscreteCodebookAssignemnt(object):
    """
    Parametrizes a discrete codebook distribution.
    """

    def __init__(
        self,
        parameters: torch.Tensor,
        codebook: nn.Embedding,
        clamp_output: bool = False,
        act_fn: str = 'none',
    ):

        self.parameters = parameters
        if clamp_output:
            parameters = torch.clamp(parameters, -5.0, 5.0)
        self.clamp_output = clamp_output
        self.act_fn = act_fn
        self.straight_through = True
        self.temp = 1.0
        self.codebook = codebook
        self.num_tokens = codebook.weight.shape[0]
        assert self.num_tokens == 128
        assert self.parameters.shape[1] == self.num_tokens

    def to_range(self, x, act_fn):
        if act_fn == 'sigmoid':
            return 2 * F.sigmoid(x) - 1
        elif act_fn == 'tanh':
            return F.tanh(x)
        elif act_fn == 'clip':
            return torch.clamp(x, -1, 1)
        elif act_fn == 'none':
            return x
        else:
            raise NotImplementedError

    def mode(self) -> torch.FloatTensor:
        indices = self.get_codebook_indices()
        # one_hot = torch.scatter(torch.zeros_like(self.parameters), 1, indices[:, None], 1.0)
        one_hot = F.one_hot(indices, num_classes=self.num_tokens).permute(0, 3, 1, 2).float()
        sampled = torch.einsum('b n h w, n d -> b d h w', one_hot, self.codebook.weight)
        return sampled

    def get_codebook_indices(self) -> torch.LongTensor:
        return self.parameters.argmax(dim=1)

    def get_codebook_probs(self) -> torch.FloatTensor:
        return nn.Softmax(dim=1)(self.parameters)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        _, indices = self.parameters.max(dim=1)
        y_hard = F.one_hot(indices, num_classes=self.num_tokens).permute(0, 3, 1, 2).float()
        y_hard = (y_hard - self.parameters).detach() + self.parameters
        sampled = torch.einsum('b n h w, n d -> b d h w', y_hard, self.codebook.weight)
        return sampled

    def kl(self):
        logits = rearrange(self.parameters, 'b n h w -> b (h w) n')
        qy = F.softmax(logits, dim=-1)
        log_qy = torch.log(qy + 1e-10)
        log_uniform = torch.log(torch.tensor([1. / self.num_tokens], device=log_qy.device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)
        return kl_div

    def get_range(self):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"DiscreteCodeBookAssignment(mean={self.parameters}, " \
               f"clamp_output={self.clamp_output}, act_fn={self.act_fn})"
