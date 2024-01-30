"""
Author: Wouter Van Gansbeke

File with UNet models for latent diffusion training
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Dict, Any, Iterable
from ldmseg.utils import OutputDict
try:
    from diffusers import UNet2DConditionModel
    from diffusers.training_utils import EMAModel
except ImportError:
    print("Diffusers package not found to train UNet. Please install diffusers")
    raise ImportError


class UNetOutput(OutputDict):
    sample: torch.FloatTensor


class UNet(UNet2DConditionModel):

    def define_dropout(self, dropout: float = 0.0, mode: str = 'standard') -> None:
        if dropout <= 0.0:
            return

        print(f'Adding dropout of {dropout} with mode {mode} ...')
        if mode == 'standard':
            self.input_dropout = nn.Dropout(dropout)
        elif mode == 'gaussian':
            self.input_dropout = GaussianDropout(dropout)
        else:
            raise NotImplementedError

    def define_learnable_embedding(self, in_channels, out_channels):
        assert self.encoder_hid_proj is None
        self.object_queries = nn.Embedding(in_channels, out_channels)

    def define_separate_encoder(self, add_adaptor: bool = False, init_mode_adaptor: str = "random"):
        from copy import deepcopy

        # take a deepcopy of the first layers to avoid sharing weights
        self.conv_in_img = deepcopy(self.conv_in)
        self.down_blocks_additional = nn.ModuleList([deepcopy(down_block) for down_block in self.down_blocks])

        if add_adaptor:
            print(f'Adding adaptor layers with init mode {init_mode_adaptor} ...')
            # add a conv layer to adapt the number of channels
            self.adaptor_layers = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1, bias=True)
                for out_channels in self.config.block_out_channels])

            # init adaptor layers using init_mode_img
            for adaptor_layer in self.adaptor_layers:
                if init_mode_adaptor == 'zero':
                    adaptor_layer.weight.data.zero_()
                    adaptor_layer.bias.data.zero_()
                else:
                    pass

    def define_upscaler(self, num_classes: int = 128, norm_num_groups: int = 32, dim: int = 256) -> None:
        conv_out = self.conv_out
        in_channels = conv_out.in_channels

        upscaler = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            LayerNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(norm_num_groups, dim),
            nn.SiLU(),
            nn.Conv2d(dim, num_classes, 3, padding=1),
        )

        self.conv_out = upscaler
        print('Successfully added upscaler to UNet ...')

    def remove_cross_attention(self):
        # down blocks
        for down_block in self.down_blocks:
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                for attn_block in down_block.attentions:
                    for transf_block in attn_block.transformer_blocks:
                        transf_block.attn2 = None
                        transf_block.norm2 = None

        # mid block
        if hasattr(self, "mid_block"):
            for attn_block in self.mid_block.attentions:
                for transf_block in attn_block.transformer_blocks:
                    transf_block.attn2 = None
                    transf_block.norm2 = None

        # up blocks
        for up_block in self.up_blocks:
            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                for attn_block in up_block.attentions:
                    for transf_block in attn_block.transformer_blocks:
                        transf_block.attn2 = None
                        transf_block.norm2 = None

    def get_lr_func(self, name: str, lr_decay_rate: float = 1.0) -> float:
        # remove prefix
        prefix = "module."
        if name.startswith(prefix):
            name = name[len(prefix):]

        # get factor
        factor = 1.0
        if name.startswith("conv_in."):
            factor = lr_decay_rate
        elif name.startswith("down_blocks."):
            factor = lr_decay_rate
        return factor

    def modify_encoder_hidden_state_proj(self, in_channels: int, out_channels: int) -> None:
        self.encoder_hid_proj = nn.Linear(in_channels, out_channels)

    def modify_encoder(
        self,
        in_channels: int = 4,
        init_mode_seg: str = "copy",
        init_mode_image: str = "copy",
        cond_channels: int = 0,
        init_mode_cond: str = "zero",
        separate_conv: bool = False,
        separate_encoder: bool = False,
        add_adaptor: bool = False,
        init_mode_adaptor: str = "random",
    ) -> None:

        assert in_channels in [4, 8], "in_channels must be 4 or 8"
        assert separate_conv + separate_encoder <= 1, "separate_conv and separate_encoder cannot both be True"

        if separate_conv:
            # we will use a different conv for segmentation and image
            self.conv_in_seg = nn.Conv2d(4, self.conv_in.out_channels, kernel_size=3, stride=1, padding=1, bias=True)

            # handle seg part
            if init_mode_seg == "zero":
                self.conv_in_seg.weight.data.zero_()
                self.conv_in_seg.bias.data.zero_()
            elif init_mode_seg == "random":
                pass
            elif init_mode_seg == "copy":
                self.conv_in_seg.weight.data.copy_(self.conv_in.weight.data)
                self.conv_in_seg.bias.data.copy_(self.conv_in.bias.data)
            elif init_mode_seg == "mean":
                self.conv_in_seg.weight.data.copy_(torch.mean(self.conv_in.weight.data, dim=1, keepdim=True).repeat(1, 4, 1, 1))  # noqa
                self.conv_in_seg.bias.data.copy_(self.conv_in.bias.data)
            elif init_mode_seg == "div":
                self.conv_in_seg.weight.data.copy_(self.conv_in.weight.data) / 2.
                self.conv_in_seg.bias.data.copy_(self.conv_in.bias.data)
            else:
                raise NotImplementedError(f"init_mode seg {init_mode_seg} not implemented")

            # handle image part
            if init_mode_image == "zero":
                self.conv_in.weight.data.zero_()
                self.conv_in.bias.data.zero_()
            elif init_mode_image == "copy":
                self.conv_in.weight.data.copy_(self.conv_in.weight.data)
                self.conv_in.bias.data.copy_(self.conv_in.bias.data)
            elif init_mode_image == "div":
                self.conv_in.weight.data.copy_(self.conv_in.weight.data) / 2.
                self.conv_in.bias.data.copy_(self.conv_in.bias.data)
            else:
                raise NotImplementedError(f"init_mode seg {init_mode_image} not implemented")

        elif separate_encoder:
            self.define_separate_encoder(add_adaptor=add_adaptor, init_mode_adaptor=init_mode_adaptor)

        elif in_channels == 8:
            # get out_channels, kernelsize, stride, padding, bias
            out_channels, kernel_size = self.conv_in.out_channels, self.conv_in.kernel_size,
            stride, padding, bias = self.conv_in.stride, self.conv_in.padding, self.conv_in.bias
            self.new_conv = nn.Conv2d(in_channels+cond_channels, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias is not None)
            # handle semseg part
            if init_mode_seg == "copy":
                self.new_conv.weight.data[:, :4].copy_(self.conv_in.weight.data)  # semseg
            elif init_mode_seg == "div":
                self.new_conv.weight.data[:, :4].copy_(self.conv_in.weight.data) / 2.  # semseg
            elif init_mode_seg == "mean":
                self.new_conv.weight.data[:, :4].copy_(torch.mean(self.conv_in.weight.data, dim=1, keepdim=True).repeat(1, 4, 1, 1))  # noqa
            elif init_mode_seg == "zero":
                self.new_conv.weight.data[:, :4].zero_()  # semseg
            elif init_mode_seg == "random":
                pass
            else:
                raise NotImplementedError(f"init_mode seg {init_mode_seg} not implemented")

            # handle image part
            if init_mode_image == "copy":
                self.new_conv.weight.data[:, 4:8].copy_(self.conv_in.weight.data)
            elif init_mode_image == "div":
                self.new_conv.weight.data[:, 4:8].copy_(self.conv_in.weight.data) / 2.
            elif init_mode_image == "mean":
                self.new_conv.weight.data[:, 4:8].copy_(torch.mean(self.conv_in.weight.data, dim=1, keepdim=True).repeat(1, 4, 1, 1))  # noqa
            elif init_mode_image == "zero":
                self.new_conv.weight.data[:, 4:8].zero_()  # semseg
            elif init_mode_image == "random":
                pass
            else:
                raise NotImplementedError(f"init_mode seg {init_mode_image} not implemented")

            # handle bias
            self.new_conv.bias.data.copy_(self.conv_in.bias.data)

            # assert statements
            assert self.new_conv.weight.data.shape == torch.Size([320, 8 + cond_channels, 3, 3])
            if init_mode_seg == "copy":
                assert torch.all(self.new_conv.weight.data[:, :4] == self.conv_in.weight.data)
            if init_mode_image == "copy":
                assert torch.all(self.new_conv.weight.data[:, 4:8] == self.conv_in.weight.data)

            if cond_channels > 0:
                if init_mode_cond == "zero":
                    self.new_conv.weight.data[:, 8:].zero_()
                elif init_mode_image == "mean":
                    self.new_conv.weight.data[:, 8:].copy_(torch.mean(self.conv_in.weight.data, dim=1, keepdim=True).repeat(1, 4, 1, 1))  # noqa
                elif init_mode_cond == "random":
                    pass
                else:
                    raise NotImplementedError(f"init_mode cond {init_mode_cond} not implemented")

            # replace first conv layer
            self.conv_in = self.new_conv

    def freeze_layers(self, layers: Tuple[str] = ['norm', 'time_embedding']) -> None:
        for layer in layers:
            if layer == 'norm':
                self.freeze_norm_layers()
            elif layer == 'time_embedding':
                self.freeze_time_embedding()
            elif layer == 'conv_in':
                self.freeze_conv_in()
            elif layer == 'down_blocks':
                self.freeze_down_blocks()
            else:
                raise NotImplementedError(f"layer {layer} not implemented")

    def freeze_norm_layers(self):
        print('Freezing norm layers ...')
        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        for m in self.modules():
            if isinstance(m, norm_module_types):
                m.requires_grad_(False)

    def freeze_time_embedding(self):
        print('Freezing time embedding ...')
        self.time_embedding.requires_grad_(False)

    def freeze_conv_in(self):
        if hasattr(self, 'conv_in_img'):
            print('Freezing conv_in layers ...')
            self.conv_in_img.requires_grad_(False)

    def freeze_down_blocks(self):
        if hasattr(self, 'conv_in_img'):
            print('Freezing down blocks ...')
            for down_block in self.down_blocks_additional:
                down_block.requires_grad_(False)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        timestep_img: Optional[Union[torch.Tensor, float, int]] = None,
    ) -> Union[UNetOutput, Tuple]:
        # Taken from diffusers library with additional changes to handle segmentation and rgb data

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        # 1. time
        timesteps = timestep
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if hasattr(self, 'down_blocks_additional'):
            if timestep_img is None:
                timesteps_img = torch.zeros_like(timesteps)
            else:
                timesteps_img = timestep_img.expand(sample.shape[0])
            t_emb_img = self.time_proj(timesteps_img)
            t_emb_img = t_emb_img.to(dtype=self.dtype)
            emb_img = self.time_embedding(t_emb_img, timestep_cond)

        # 2. pre-process
        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        if hasattr(self, 'object_queries'):
            encoder_hidden_states = self.object_queries.weight.unsqueeze(0).repeat(sample.shape[0], 1, 1)

        if hasattr(self, 'input_dropout'):
            sample = self.input_dropout(sample)

        # 3a. handle segmentation + image fusion and down blocks (image)
        if hasattr(self, 'down_blocks_additional'):
            sample_seg, sample_img = sample.chunk(2, dim=1)

            sample_img = self.conv_in_img(sample_img)
            down_block_additional_residuals = (sample_img,)
            for block_idx, downsample_block_img in enumerate(self.down_blocks_additional):
                if hasattr(downsample_block_img, "has_cross_attention") and downsample_block_img.has_cross_attention:
                    sample_img, res_samples = downsample_block_img(
                        hidden_states=sample_img,
                        temb=emb_img,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample_img, res_samples = downsample_block_img(hidden_states=sample_img, temb=emb_img)

                if hasattr(self, 'adaptor_layers'):
                    res_samples = tuple([self.adaptor_layers[block_idx](int_feature) for int_feature in res_samples])

                down_block_additional_residuals += res_samples

            sample = self.conv_in(sample_seg)
        elif hasattr(self, 'conv_in_seg'):
            assert sample.shape[1] == 8, "sample should have 8 channels"
            sample_seg, sample_img = sample.chunk(2, dim=1)
            sample = self.conv_in_seg(sample_seg) + self.conv_in(sample_img)
        else:
            sample = self.conv_in(sample)

        # 3b. down blocks segmentation
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. potentially add image residuals
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 5. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 6. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 7. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNetOutput(sample=sample)


# EMA Model to save GPU memory
# TODO: Move helper functionality to a separate file
# Overwrite step method in EMAmodel class to use CPU
# around 2.5x slower than GPU version
class EMAModelCPU(EMAModel):

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        parameters = list(parameters)
        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - param.cpu()))
            else:
                s_param.copy_(param.cpu())


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


class GaussianDropout(nn.Module):
    def __init__(self, prob: float):
        super(GaussianDropout, self).__init__()
        self.prob = prob / (1 - prob)
        assert 0 <= self.prob < 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = torch.ones_like(x)
            std = (self.prob / (1.0 - self.prob))**0.5
            eps = torch.normal(mean=mean, std=std)
            return x * eps
        else:
            return x
