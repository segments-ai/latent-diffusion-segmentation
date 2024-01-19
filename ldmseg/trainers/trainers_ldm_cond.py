"""
Author: Wouter Van Gansbeke

File for training a diffusion model for segmentation
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import math
import time
import wandb
from pathlib import Path
from datetime import timedelta
from termcolor import colored
from typing import Callable, List, Dict, Any, Optional, Tuple, Union, Iterable
from tqdm import tqdm
from PIL import Image
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from .optim import get_optim_unet
from ldmseg.data.dataset_base import DatasetBase
from ldmseg.utils import (
    AverageMeter, ProgressMeter, OutputDict,
    cosine_scheduler, warmup_scheduler, step_scheduler,
    is_main_process, color_map, gpu_gather, collate_fn,
    get_imagenet_stats, get_world_size
)


class ModelInputs(OutputDict):
    images: torch.Tensor
    rgb_images: torch.Tensor
    latents: torch.Tensor
    rgb_latents: torch.Tensor
    encoder_hidden_states: torch.Tensor
    original_latents: torch.Tensor
    loss_mask: torch.Tensor
    text: List[str]
    inpainting_masks: Optional[torch.Tensor] = None
    dropout: Optional[float] = None


class TrainerDiffusion(DatasetBase):

    def __init__(
        self,
        p: Dict[str, Any],
        vae_image: nn.Module,
        vae_semseg: nn.Module,
        unet_model: nn.Module,
        tokenizer: nn.Module,
        textencoder: nn.Module,
        noise_scheduler: nn.Module,
        *,
        image_descriptor_model: Optional[nn.Module] = None,
        ema_unet: Optional[nn.Module] = None,
        ema_on: bool = False,
        save_and_sample_every: int = 1000,
        results_folder: str = './results',
        args: Dict[str, Any] = None,
        cudnn_on: bool = True,
        fp16: Optional[bool] = False,
        weight_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Trainer class for diffusion models
        This class implements the training loop and evaluation

        args:
            p (Dict[str, Any]): dictionary containing the parameters
            vae_image (nn.Module): image encoder (decoder is removed)
            vae_semseg (nn.Module): segmentation vae (encoder & decoder net)
            unet_model (nn.Module): diffusion model (UNet)
            tokenizer (nn.Module): tokenizer to use
            textencoder (nn.Module): text encoder (e.g. CLIP)
            noise_scheduler (nn.Module): noise scheduler (e.g. DDIM)
            image_descriptor_model (nn.Module): image descriptor model (e.g. DINO)
            ema_unet (nn.Module): exponential moving average of the UNet
            ema_on (bool): use ema or not
            save_and_sample_every (int): save and sample every x steps
            results_folder (str): folder to save results
            args (Any): additional distributed arguments
            cudnn_on (bool): use cudnn or not
            fp16 (bool): use fp16 or not
            weight_dtype (torch.dtype): weight dtype (e.g., torch.float32)
        """

        # init inherited class
        super(TrainerDiffusion, self).__init__(data_dir=p['data_dir'])

        # save arguments
        self.p = p
        self.args = args

        # handle fp16 scaler
        self.fp16_scaler = torch.cuda.amp.GradScaler() if fp16 else None
        if fp16:
            print(colored('Warning -- Using FP16', 'yellow'))

        # model
        self.vae_image = vae_image
        self.vae_semseg = vae_semseg
        self.unet_model = unet_model
        self.tokenizer = tokenizer
        self.textencoder = textencoder
        self.noise_scheduler = noise_scheduler
        self.image_descriptor_model = image_descriptor_model
        self.vae_image.requires_grad_(False)
        self.vae_semseg.requires_grad_(False)
        self.vae_image.eval()   # has only effect on dropout (groupnorm is not affected)
        self.vae_semseg.eval()  # has only effect on dropout (groupnorm is not affected)
        self.freeze_layers = self.p['train_kwargs']['freeze_layers']
        if self.image_descriptor_model is not None:
            self.image_descriptor_model.requires_grad_(False)
            self.image_descriptor_model.eval()
        if self.textencoder is not None:
            assert self.image_descriptor_model is None
            self.textencoder.requires_grad_(False)
            self.textencoder.eval()
        self.decoder = None

        # dtype
        self.weight_dtype = weight_dtype if weight_dtype is not None else torch.float32

        self.clip_grad = p['train_kwargs']['clip_grad']
        if self.clip_grad > 0:
            print(colored(f'Warning -- Using gradient clipping of {self.clip_grad}', 'yellow'))

        # set parameters
        self.use_wandb = p['wandb']
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = p['train_kwargs']['batch_size']
        self.batch_size_val = min(p['eval_kwargs']['batch_size'], self.batch_size)
        self.num_workers = p['train_kwargs']['num_workers']
        self.gradient_accumulate_every = p['train_kwargs']['accumulate']
        assert self.gradient_accumulate_every >= 1 and isinstance(self.gradient_accumulate_every, int)
        self.train_num_steps = p['train_kwargs']['train_num_steps']
        self.print_freq = p['eval_kwargs']['print_freq']
        assert self.print_freq % self.gradient_accumulate_every == 0
        if self.gradient_accumulate_every > 1:
            print(colored('Warning -- Accumulating gradients', 'yellow'))
        self.eff_batch_size = self.batch_size * self.gradient_accumulate_every
        self.use_ema = ema_on
        self.image_size = p['transformation_kwargs']['size']
        self.image_size_max = p['transformation_kwargs']['max_size']
        if 'max_edge' in p['transformation_kwargs']['type']:
            self.image_size = self.image_size_max
        self.rgb_size = p['transformation_kwargs']['size_rgb']
        self.latent_size = self.image_size // self.vae_semseg.downsample_factor
        self.training_loss_type = p['train_kwargs']['loss']
        self.ohem_ratio = self.p['train_kwargs']['ohem_ratio']
        self.sample_posterior = self.p['train_kwargs']['sample_posterior']
        self.sample_posterior_rgb = self.p['train_kwargs']['sample_posterior_rgb']
        self.prob_train_on_pred = self.p['train_kwargs']['prob_train_on_pred']
        self.prob_inpainting = self.p['train_kwargs']['prob_inpainting']
        self.rgb_noise_level = self.p['train_kwargs']['rgb_noise_level']
        self.min_noise_level = self.p['train_kwargs']['min_noise_level']
        self.cond_noise_level = self.p['train_kwargs']['cond_noise_level']
        self.type_mask = self.p['train_kwargs']['type_mask']
        self.sampling_kwargs = p['sampling_kwargs']
        self.cmap = color_map()
        self.mask_th = p['eval_kwargs']['mask_th']
        self.count_th = p['eval_kwargs']['count_th']
        self.overlap_th = p['eval_kwargs']['overlap_th']
        self.dropout = p['train_kwargs']['dropout']
        self.self_condition = p['train_kwargs']['self_condition']
        self.num_inference_steps = self.sampling_kwargs['num_inference_steps']

        # optimizer
        if isinstance(self.unet_model, nn.parallel.DistributedDataParallel):
            self.unet_dtype = self.unet_model.module.dtype
            lr_func_model = self.unet_model.module.get_lr_func
        else:
            self.unet_dtype = self.unet_model.dtype
            lr_func_model = self.unet_model.get_lr_func
        lr_factor_func = partial(
            lr_func_model,
            lr_decay_rate=p['optimizer_backbone_multiplier'])
        self.opt, self.save_optim = get_optim_unet(
            self.unet_model,
            base_lr=p['optimizer_kwargs']['lr'],
            weight_decay=p['optimizer_kwargs']['weight_decay'],
            weight_decay_norm=p['optimizer_kwargs']['weight_decay_norm'],
            betas=p['optimizer_kwargs']['betas'],            # adam beta1, beta2
            lr_factor_func=lr_factor_func,                   # lr decay for backbone
            zero_redundancy=p['optimizer_zero_redundancy'],  # zero redundancy in optim in DDP
            save_optim=p['optimizer_save_optim'],            # save optimizer state or not
            verbose=False,                                   # print all parameters if true
        )
        self.lr = self.opt.param_groups[0]['lr']
        print(self.opt)

        # cudnn / cuda
        cudnn.benchmark = cudnn_on
        torch.backends.cuda.matmul.allow_tf32 = p['train_kwargs']['allow_tf32']

        # dataset and dataloader
        self.transforms = self.get_train_transforms(p['transformation_kwargs'])
        self.transforms_val = self.get_val_transforms(p['transformation_kwargs'])
        print('Train', self.transforms)
        print('Eval', self.transforms_val)
        self.ds = self.get_dataset(
            p['train_db_name'],
            transform=self.transforms,
            tokenizer=self.tokenizer,
            split=p['split'],
            remap_labels=p['train_kwargs']['remap_seg'],
            caption_dropout=p['train_kwargs']['caption_dropout'],
            caption_type=p['train_kwargs']['caption_type'],
            encoding_mode=p['train_kwargs']['encoding_mode'],
            inpaint_mask_size=p['train_kwargs']['inpaint_mask_size'],
            num_classes=p['num_classes'],
            fill_value=p['fill_value'],
            ignore_label=p['ignore_label'],
            inpainting_strength=p['inpainting_strength'],
        )
        self.ds_val = self.get_dataset(
            p['val_db_name'],
            split='val',
            transform=self.transforms_val,
            tokenizer=self.tokenizer,
            caption_dropout=1.0,
            remap_labels=p['train_kwargs']['remap_seg'],
            caption_type=p['train_kwargs']['caption_type'],
            encoding_mode=p['train_kwargs']['encoding_mode'],
            num_classes=p['num_classes'],
            fill_value=p['fill_value'],
            ignore_label=p['ignore_label'],
            inpainting_strength=p['inpainting_strength'],
        )

        print(colored('The dataset contains {} samples'.format(len(self.ds)), 'blue'))
        if args['distributed']:
            self.train_sampler = DistributedSampler(self.ds)
            self.val_sampler = DistributedSampler(self.ds_val)
        else:
            self.train_sampler = None
            self.val_sampler = None
        # train loader
        self.dl = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        # val loader
        self.dl_val = DataLoader(
            self.ds_val,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=self.val_sampler,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # ema
        if self.use_ema:
            self.ema_unet_model = ema_unet

        # save dir
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.visualization_dir = 'visualizations'
        os.makedirs(self.visualization_dir, exist_ok=True)
        print('Images will be saved to {}'.format(self.visualization_dir))

        # step counter state
        self.step = 0
        self.start_epoch = 0
        self.num_iters_per_epoch = math.ceil(len(self.dl) / self.gradient_accumulate_every)
        self.epochs = math.ceil(self.train_num_steps / self.num_iters_per_epoch)
        self.total_iters = self.epochs * self.num_iters_per_epoch
        print(colored(f'training for {self.epochs} epochs or {self.num_iters_per_epoch} iters per epoch '
                      f'or {self.total_iters} iterations',
                      'yellow'))

        # learning rate scheduler
        self.lr_scheduler = None
        if 'lr_scheduler_name' in self.p.keys():
            try:
                self.lr_scheduler = self.get_lr_scheduler(p['lr_scheduler_name'], **p['lr_scheduler_kwargs'])
            except NotImplementedError:
                print(colored('Warning -- No learning rate scheduler found', 'yellow'))
                assert self.lr_scheduler is None

        # imagenet normalization
        stats = get_imagenet_stats()
        self.pixel_mean_in = stats['mean'].to(self.unet_model.device)
        self.pixel_std_in = stats['std'].to(self.unet_model.device)
        self.pixel_mean_clip = stats['mean_clip'].to(self.unet_model.device)
        self.pixel_std_clip = stats['std_clip'].to(self.unet_model.device)

        # best model
        self.best_pq = 0.0

        # print info
        print('min_noise_level: {}'.format(self.min_noise_level))
        print('rgb_noise_level: {}'.format(self.rgb_noise_level))
        print('sample_posterior: {}'.format(self.sample_posterior))
        print('sample_posterior_rgb: {}'.format(self.sample_posterior_rgb))
        print('prob_train_on_pred: {}'.format(self.prob_train_on_pred))
        print('prob inpainting: {}'.format(self.prob_inpainting))
        print('image descriptor model: {}'.format(self.image_descriptor_model))
        print('type mask: {}'.format(self.type_mask))
        print(f'SIZE SEG: {self.image_size}x{self.image_size}, SIZE RGB {self.rgb_size}x{self.rgb_size}')

    def encode_seg(self, semseg, cmap=None):
        # we will encode the semseg map with a fixed color map
        if cmap is None:
            cmap = color_map()
        seg_t = semseg.astype(np.uint8)
        array_seg_t = np.empty((seg_t.shape[0], seg_t.shape[1], seg_t.shape[2], cmap.shape[1]), dtype=cmap.dtype)
        for class_i in np.unique(seg_t):
            array_seg_t[seg_t == class_i] = cmap[class_i]
        return array_seg_t

    @torch.no_grad()
    def encode_inputs(
        self,
        images: torch.tensor,
        sample_posterior: bool = False,
        encode_func: Optional[Union[Callable, nn.Module]] = None,
        scaling_factor: Optional[float] = None,
        resize: Optional[int] = None,
        weight_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Encode images to latents using the VAE or AE
        This funtion can handle both RGB and segmentation images
        A different scaling factor can be used as well as encoder

        args:
            images (torch.Tensor): images to encode
            sample_posterior (bool): sample from posterior or use mode/mean
            encode_func (nn.Module): encoder to use
            scaling_factor (float): scaling factor to use
            resize (int): resize input images to a different size (to save memory)

        returns:
            latents (torch.Tensor): latents
        """

        if encode_func is None:
            encode_func = self.vae_image.encode
        if scaling_factor is None:
            scaling_factor = self.vae_image.scaling_factor
        if resize is not None:
            images = F.interpolate(images, size=(resize, resize), mode='bilinear', align_corners=False)
        if weight_dtype is None:
            weight_dtype = self.weight_dtype

        images = 2. * images - 1.
        if sample_posterior:
            latent_dist = encode_func(images.to(weight_dtype)).latent_dist
            latents_mean = latent_dist.mode()
            latents = latent_dist.sample()
        else:
            latents = encode_func(images.to(weight_dtype)).latent_dist.mode()
            latents_mean = latents.clone()

        if resize is not None:
            latents = F.interpolate(
                latents,
                size=(self.latent_size, self.latent_size),
                mode='bilinear',
                align_corners=False
            )
            latents_mean = F.interpolate(
                latents_mean,
                size=(self.latent_size, self.latent_size),
                mode='bilinear',
                align_corners=False
            )

        latents = latents * scaling_factor
        latents_mean = latents_mean * scaling_factor
        return latents, latents_mean

    @torch.no_grad()
    def decode_latents(
        self,
        latents: torch.tensor,
        return_logits: bool = False,
        threshold_output: bool = False,
        rgb_latents: Optional[torch.Tensor] = None,
        weight_dtype: torch.dtype = torch.float32,
    ) -> np.ndarray:
        """
        Decode latents to images using the VAE or AE
        This funtion only handles segmentation images

        args:
            latents (torch.Tensor): latents to decode
            cmap (np.ndarray): color map to use
            rgb_latents (torch.Tensor): rgb latents to use
            return_logits (bool): return logits or not

        returns:
            images (np.ndarray): decoded images
        """
        if weight_dtype is None:
            weight_dtype = self.weight_dtype

        latents = latents * (1. / self.vae_semseg.scaling_factor)
        images = self.vae_semseg.decode(latents.to(weight_dtype))
        images = images.float()
        if return_logits:
            return images

        if images.shape[1] != 3:
            predictions = torch.argmax(images, dim=1)

            if threshold_output:
                probs = F.softmax(images, dim=1)
                probs = probs.max(dim=1)[0]
                predictions[probs < self.mask_th] = self.ds.ignore_label

            predictions = predictions.cpu().numpy()
            images = self.encode_seg(predictions).astype(np.uint8)
        else:
            assert images.shape[1] == 3
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).astype(np.uint8)
        return images

    @torch.no_grad()
    def predict_sample(
        self,
        latents: torch.Tensor,
        rgb_latents: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        tmax: Optional[int] = None,
    ):
        """
        Predict the original latents from noisy latents with the diffusion model

        args:
            latents (torch.Tensor): latents to predict
            rgb_latents (torch.Tensor): rgb latents to use
            encoder_hidden_states (torch.Tensor): hidden states to use
            tmax (int): maximum number of timesteps to use

        returns:
            pred_latents (torch.Tensor): predicted latents
        """

        if tmax is None:
            tmax = self.noise_scheduler.num_train_timesteps

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, tmax, (latents.shape[0],), device=latents.device,
            dtype=torch.long,
        )

        # add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        if rgb_latents is None:
            inputs = noisy_latents
        else:
            inputs = torch.cat([noisy_latents, rgb_latents], dim=1)

        # forward pass
        prediction = self.unet_model(inputs, timesteps, encoder_hidden_states).sample
        if self.noise_scheduler.prediction_type == 'epsilon':
            pred_latents = self.noise_scheduler.remove_noise(
                noisy_latents.detach(), prediction.detach(), timesteps)
        elif self.noise_scheduler.prediction_type == 'sample':
            pred_latents = prediction
        else:
            raise ValueError('Unknown prediction type: {}'.format(self.noise_scheduler.prediction_type))

        # clamp
        pred_latents = torch.clamp(pred_latents, latents.min(), latents.max())
        return pred_latents

    def loss_fn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        reduction: str = 'none',
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """ Compute the loss between two tensors

        args:
            x (torch.Tensor): tensor 1
            y (torch.Tensor): tensor 2
            reduction (str): reduction type
            mask (torch.Tensor): mask to use after loss

        returns:
            losses (torch.Tensor): losses
        """

        if self.training_loss_type == 'l1':
            losses = F.l1_loss(x, y, reduction=reduction)
        elif self.training_loss_type == 'l2':
            losses = F.mse_loss(x, y, reduction=reduction)
        elif self.training_loss_type == 'smooth_l1':
            losses = F.smooth_l1_loss(x, y, reduction=reduction)
        else:
            raise ValueError('Unknown loss type: {}'.format(self.training_loss_type))

        if mask is not None:
            losses = losses * mask[:, None]  # multiply loss with a weigth mask
        return losses

    def compute_loss(
        self,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        noisy_latents: torch.Tensor,
        rgb_latents: Optional[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        loss_mask: Optional[torch.Tensor],
        latents: Optional[torch.Tensor] = None,
        original_latents: Optional[torch.Tensor] = None,
        inpainting_masks: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Loss is computed as follows:
        1. add noise to the latents according to the noise scheduler at each timestep
        2. predict the added noise of the latents (or equivalently the original latents)
        3. compute the MSE loss between the added noise and the predicted noise (or original latents)

        args:
            latents (torch.Tensor): latents to predict
            rgb_latents (torch.Tensor): rgb latents to use
            encoder_hidden_states (torch.Tensor): hidden states to use
            loss_mask (torch.Tensor): loss mask to use in weighted loss
            original_latents (torch.Tensor): original latents to use in case of `train_on_pred` is True

        returns:
            loss (torch.Tensor): loss
            noisy_latents (torch.Tensor): latents with added noise
            pred_latents (torch.Tensor): predicted latents (sample)
            timesteps (torch.Tensor): timestep at which the noise was added
        """
        assert rgb_latents is not None

        if self.self_condition and condition is None:
            condition = torch.zeros_like(noisy_latents)

        # 1a. (optional) add rgb noise
        timesteps_img = None
        if self.rgb_noise_level > 0:
            rgb_noise = torch.randn_like(rgb_latents)
            timesteps_img = torch.randint(
                0, self.rgb_noise_level, (rgb_latents.shape[0],), device=rgb_latents.device, dtype=torch.long)
            rgb_latents = self.noise_scheduler.add_noise(rgb_latents, rgb_noise, timesteps_img)
        inputs = torch.cat([noisy_latents, rgb_latents], dim=1)

        # 1b. (optional) add conditioning with noise
        if condition is not None:
            if self.cond_noise_level > 0:
                cond_noise = torch.randn_like(condition)
                timesteps_cond = torch.randint(
                    0, self.cond_noise_level, (condition.shape[0],), device=condition.device, dtype=torch.long)
                condition = self.noise_scheduler.add_noise(condition, cond_noise, timesteps_cond)
            inputs = torch.cat([inputs, condition], dim=1)

        # 2. predict the added noise of the latents
        if self.noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.prediction_type == "sample":
            target = original_latents if original_latents is not None else latents
        else:
            raise ValueError(f"Unknown prediction type: {self.noise_scheduler.prediction_type}")
        prediction = self.unet_model(inputs, timesteps, encoder_hidden_states, timestep_img=timesteps_img).sample

        # 3. minimize MSE loss between the added noise and the predicted noise
        loss = self.loss_fn(prediction.float(), target.float(), reduction='none', mask=loss_mask)

        if hasattr(self.noise_scheduler, 'weights'):
            loss = loss * self.noise_scheduler.weights[timesteps][:, None, None, None]

        loss = loss.view(-1)
        if self.ohem_ratio < 1.:
            loss = torch.topk(loss, int(self.ohem_ratio * loss.numel()))[0]
        loss = loss.mean()

        # (optional) remove noise from the latents for visualization later on
        if self.noise_scheduler.prediction_type == 'epsilon':
            pred_latents = self.noise_scheduler.remove_noise(
                noisy_latents.detach(), prediction.detach(), timesteps)
        elif self.noise_scheduler.prediction_type == 'sample':
            pred_latents = prediction
        else:
            raise ValueError(f"Unknown prediction type: {self.noise_scheduler.prediction_type}")

        # (optional) paste the original latents according to the inpainting mask
        if inpainting_masks is not None:
            pred_latents[inpainting_masks] = original_latents[inpainting_masks]

        return loss, noisy_latents.detach(), pred_latents.detach(), timesteps

    @torch.no_grad()
    def get_loss_weight_mask(
        self,
        targets: torch.Tensor,
        ignore_label: int = 0,
        mode: str = 'nearest',
        size: Tuple[int] = (64, 64),
        device: str = 'cuda',
        type_mask: str = 'ignore',
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get loss weight mask to use in the loss function later on

        args:
            targets (torch.Tensor): targets
            ignore_label (int): ignore label
            mode (str): interpolation mode
            size (tuple): size of the mask
            device (str): device to use
            type_mask (str): type of mask to use
            padding_mask (torch.Tensor): padding mask to use
        """

        if type_mask == 'counts':
            # TODO: move this to the dataset (i.e. cpu)
            targets = F.interpolate(targets[:, None].float(), size=size, mode=mode).squeeze(1)
            mask = torch.zeros_like(targets, device=device, dtype=torch.float32)
            for idx_t, target in enumerate(targets):
                unique_classes, counts = torch.unique(target, return_counts=True)
                for class_i, c_i in zip(unique_classes, counts):
                    if class_i == ignore_label:
                        continue
                    mask[idx_t, target == class_i] = 1. / c_i
        elif type_mask == 'ignore':
            targets = F.interpolate(targets[:, None].float(), size=size, mode=mode).squeeze(1)
            mask = (targets != self.ds.ignore_label).to(device=device, dtype=torch.float32)
        elif type_mask == 'padding':
            padding_mask = F.interpolate(padding_mask[:, None].float(), size=size, mode=mode).squeeze(1)
            mask = padding_mask.to(device=device, dtype=torch.float32)  # sets padding to 0 in loss
        else:
            mask = None
        return mask

    def norm_resize_images(self, x, mode='imagenet'):
        # TODO: clean up and make more general

        identifier = self.image_descriptor_model.__class__.__name__.lower()
        if 'clip' in identifier:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x = (x - self.pixel_mean_clip) / self.pixel_std_clip
        elif 'dino' in identifier:
            x = F.interpolate(x, size=(518, 518), mode='bilinear', align_corners=False)
            x = (x - self.pixel_mean_in) / self.pixel_std_in
        else:
            x = (x - self.pixel_mean_in) / self.pixel_std_in
        return x

    def process_inputs(self, data: dict) -> ModelInputs:
        """
        Process inputs for training

        args:
            data (dict): dictionary containing the sampled data (images, text, etc.)
        """

        rgb_images = data['image'].cuda(self.args['gpu'], non_blocking=True)
        images = data['image_semseg'].cuda(self.args['gpu'], non_blocking=True)
        text = data['text']

        # get latents
        latents, latents_mean = self.encode_inputs(
            images,
            encode_func=self.vae_semseg.encode,
            scaling_factor=self.vae_semseg.scaling_factor,
            sample_posterior=self.sample_posterior,
            weight_dtype=torch.float32,  # always use float32 for segmentation
        )

        rgb_latents, _ = self.encode_inputs(
            rgb_images,
            encode_func=self.vae_image.encode,
            scaling_factor=self.vae_image.scaling_factor,
            sample_posterior=self.sample_posterior_rgb,
            resize=self.rgb_size,
        )

        latents = latents.to(torch.float32)
        latents_mean = latents_mean.to(torch.float32)
        rgb_latents = rgb_latents.to(torch.float32)

        # (optional) apply inpainting mask
        inpainting_masks = None
        if self.prob_inpainting > 0.:
            mask_batch = torch.rand((images.shape[0],), generator=None) < self.prob_inpainting
            inpainting_masks = data['inpainting_mask'].cuda(self.args['gpu'], non_blocking=True)
            inpainting_masks = F.interpolate(
                inpainting_masks[:, None].float(), size=latents.shape[-2:], mode='nearest').squeeze(1)
            inpainting_masks[~mask_batch] = 0.

        # (optional) handle image descriptors for conditioning
        encoder_hidden_states = None
        if self.image_descriptor_model is not None:
            with torch.no_grad():
                inputs_image_descriptors = self.norm_resize_images(rgb_images)
                descriptors = self.image_descriptor_model(inputs_image_descriptors.to(self.weight_dtype))['last_feat']  # noqa
                descriptors = descriptors.view(descriptors.shape[0], descriptors.shape[1], -1).permute(0, 2, 1)
                encoder_hidden_states = descriptors.to(torch.float32)

        # (optional) handle text encoder for conditioning
        if self.textencoder is not None:
            assert self.image_descriptor_model is None
            tokens = data['tokens'].cuda(self.args['gpu'], non_blocking=True)
            text_embeddings = self.textencoder(tokens)[0]
            encoder_hidden_states = text_embeddings.to(torch.float32)

        # (optional) make initial prediction
        if self.prob_train_on_pred > 0.:
            mask_batch = torch.rand((latents.shape[0],), generator=None) < self.prob_train_on_pred
            if mask_batch.sum() > 0:
                latents[mask_batch] = self.predict_sample(
                    latents[mask_batch], rgb_latents[mask_batch],
                    encoder_hidden_states[mask_batch] if encoder_hidden_states is not None else None,
                    tmax=self.noise_scheduler.num_train_timesteps // 2)  # lower tmax

        # (optional) handle ignore mask
        loss_mask = self.get_loss_weight_mask(
            data['semseg'],
            mode='nearest',
            device=self.args['gpu'],
            size=(self.latent_size, self.latent_size),
            ignore_label=self.ds.ignore_label,
            type_mask=self.type_mask,
            padding_mask=data['mask'],
        )

        return ModelInputs(
            text=text,
            rgb_images=rgb_images,
            latents=latents,
            rgb_latents=rgb_latents,
            encoder_hidden_states=encoder_hidden_states,
            original_latents=latents_mean,
            loss_mask=loss_mask,
            inpainting_masks=inpainting_masks,
            dropout=self.dropout if self.dropout > 0 else None,
        )

    def update_weights(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """ Update weights after clipping gradients
        """
        if self.fp16_scaler is None:
            if self.clip_grad > 0:
                clip_grad_norm_(parameters, self.clip_grad, norm_type=2)
            self.opt.step()
        else:
            if self.clip_grad > 0:
                self.fp16_scaler.unscale_(self.opt)  # unscale the gradients in-place
                clip_grad_norm_(parameters, self.clip_grad, norm_type=2)
            self.fp16_scaler.step(self.opt)
            self.fp16_scaler.update()

    def update_scheduler(self, batch_idx: int) -> None:
        """ Update learning rate scheduler
        """
        if self.lr_scheduler is not None:
            for param_group in self.opt.param_groups:
                param_group["lr"] = self.lr_scheduler[self.step]
            if batch_idx + 1 == self.gradient_accumulate_every:
                print(f"Learning rate is set to: {self.opt.param_groups[0]['lr']:.3e}")

    def train_single_epoch(
            self,
            epoch: int,
            losses: AverageMeter,
            log_dict: dict,
            progress: ProgressMeter,
    ) -> None:
        """
        Train the model for a single epoch

        args:
            epoch (int): current epoch
            losses (AverageMeter): losses
            log_dict (dict): dictionary to log
            progress (ProgressMeter): progress meter
        """

        total_loss = 0.
        for batch_idx, data in enumerate(self.dl):

            # process inputs
            model_inputs = self.process_inputs(data)

            # add noise to the latents according to the noise magnitude at each timestep
            noise = torch.randn_like(model_inputs.latents)
            timesteps = torch.randint(
                self.min_noise_level, self.noise_scheduler.num_train_timesteps,
                (model_inputs.latents.shape[0],), device=model_inputs.latents.device,
                dtype=torch.long,
            )
            noisy_latents = self.noise_scheduler.add_noise(model_inputs.latents, noise, timesteps)

            # (optional) get condition mask
            condition = None
            if self.self_condition:
                assert model_inputs.rgb_latents is not None
                condition = torch.zeros_like(noisy_latents)
                with torch.no_grad():
                    inputs = torch.cat([noisy_latents, model_inputs.rgb_latents, condition], dim=1)
                    with torch.autocast('cuda', enabled=self.fp16_scaler is not None):
                        pred = self.unet_model(inputs, timesteps, model_inputs.encoder_hidden_states).sample
                    condition = self.noise_scheduler.remove_noise(noisy_latents, pred, timesteps)

            # calculate loss
            with torch.autocast('cuda', enabled=self.fp16_scaler is not None):
                loss, noisy_latents, pred_latents, timesteps = self.compute_loss(
                    noise,
                    timesteps,
                    noisy_latents,
                    model_inputs.rgb_latents,
                    model_inputs.encoder_hidden_states,
                    loss_mask=model_inputs.loss_mask,
                    latents=model_inputs.latents,
                    original_latents=model_inputs.original_latents,
                    inpainting_masks=model_inputs.inpainting_masks,
                    condition=condition,
                )
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.detach()

            # calculate gradients
            if self.fp16_scaler is None:
                loss.backward()
            else:
                self.fp16_scaler.scale(loss).backward()

            # accumulate gradients
            if ((batch_idx + 1) % self.gradient_accumulate_every != 0) and (batch_idx + 1 != len(self.dl)):
                continue

            # update learning rate and weights
            self.update_scheduler(batch_idx)
            self.update_weights(self.unet_model.parameters())
            self.opt.zero_grad()
            dist.barrier()

            # update meters
            torch.cuda.synchronize()
            total_loss = gpu_gather(total_loss.repeat(model_inputs.latents.size(0))).mean().item()
            losses.update(total_loss, model_inputs.latents.size(0))
            total_loss = 0.0

            # update step
            self.step += 1

            # update ema
            if is_main_process() and self.use_ema:
                self.ema_unet_model.step(self.unet_model.parameters())

            # validate and logging
            if self.check_iter(batch_idx, epoch) and is_main_process():
                vis_length = self.batch_size_val
                self.log_images_train(
                    log_dict,
                    noisy_prompts=model_inputs.text[:vis_length],
                    noisy_latents=noisy_latents[:vis_length],
                    pred_latents=pred_latents[:vis_length],
                    timesteps=timesteps[:vis_length],
                    rgb_latents=model_inputs.rgb_latents[:vis_length],
                    rgb_images=model_inputs.rgb_images[:vis_length],
                    gt_images=data['semseg'][:vis_length],
                    original_latents=model_inputs.original_latents[:vis_length],
                    inpainting_masks=model_inputs.inpainting_masks[:vis_length] if model_inputs.inpainting_masks is not None else None,  # noqa
                    seed=0,
                )

            # display progress
            if (batch_idx + 1) % (self.print_freq // self.gradient_accumulate_every) == 0:
                progress.display(batch_idx)

    def train_loop(self) -> None:
        """ Train loop that trains the model for a given number of epochs
        """
        # max_iter=1024 // (self.batch_size_val * get_world_size()),  # consider only 1024 images to save time
        max_iter = None

        # define evaluation function to monitor progress
        evaluation_fn = partial(
            self.compute_metrics,
            metrics=['pq'],
            dataloader=self.dl_val,
            threshold_output=True,
            save_images=True,
            seed=0,
            max_iter=max_iter,
            models_to_eval=[self.unet_model],
            num_inference_steps=self.num_inference_steps,
            set_save_model=False,
        )

        print('Evaluating ...')
        evaluation_fn()

        # start training loop
        start_training_time = time.time()
        self.unet_model.train()
        for epoch in range(self.start_epoch, self.epochs):

            # track epoch id
            print(colored('-'*25, 'blue'))
            print(colored(f"Starting epoch {epoch}", "blue"))

            # define containers and progress meters to track loss
            log_dict = {}
            losses = AverageMeter("Loss", ":.4e")
            progress = ProgressMeter(
                len(self.dl),
                [losses],
                prefix="Epoch: [{}]".format(epoch),
            )
            self.epoch = epoch

            # randomize sampler at the start of each epoch
            if self.args['distributed']:
                self.dl.sampler.set_epoch(epoch)

            # start counting time
            start_epoch_time = time.time()

            # start looping over batches
            self.train_single_epoch(epoch, losses, log_dict, progress)

            # save model
            dist.barrier()
            if is_main_process():
                self.save(epoch)
                print(colored(f'Model saved for run {self.p["name"]}', 'yellow'))

            # log average loss at the end of each epoch
            if self.use_wandb and is_main_process():
                log_dict.update({"average_loss_epoch": losses.avg})
                wandb.log(log_dict)

            # validate
            print('Start evaluation ...')
            evaluation_fn(set_save_model=True)

            # print remaining training time
            print(colored(f'Average loss: {losses.avg:.3e}', 'yellow'))
            time_per_epoch = time.time() - start_epoch_time
            print(colored(f'Epoch took {str(timedelta(seconds=time_per_epoch))}', 'yellow'))
            avg_time_per_epoch = (time.time() - start_training_time) / (epoch + 1 - self.start_epoch)
            eta = avg_time_per_epoch * (self.epochs - 1 - epoch)
            print(colored(f'ETA: {str(timedelta(seconds=eta ))}', 'yellow'))

        print('Final evaluation on the full validation set ...')
        evaluation_fn(max_iter=None)
        print(f"Finished run {self.p['name']} and took {str(timedelta(seconds=time.time()-start_training_time))}")

    def check_iter(self, batch_idx: int, epoch: int) -> bool:
        # check when to save and sample
        return (self.step != 0 and self.step % self.save_and_sample_every == 0) or \
               (epoch == self.epochs - 1 and batch_idx == len(self.dl) - 1)

    def compute_miou(self, threshold_output: bool = False, save_images: bool = False):
        """ Compute the mIoU on the validation set using a conditioning mask """
        raise NotImplementedError

    def compute_metrics(
        self,
        metrics: Union[List[str], str] = ['pq'],
        threshold_output: bool = False,
        save_images: bool = False,
        seed: Optional[int] = None,
        max_iter: Optional[int] = None,
        dataloader: Optional[DataLoader] = None,
        models_to_eval: Optional[List[nn.Module]] = None,
        num_inference_steps: int = 50,
        set_save_model: bool = False,
        block_size: Optional[int] = None,
        prob_mask: Optional[float] = None,
    ):
        """ Compute different metrics on the validation set
        """

        assert isinstance(metrics, str) or isinstance(metrics, list)

        if not isinstance(metrics, list):
            metrics = [metrics]

        dist.barrier()

        # set models to eval
        for m in models_to_eval:
            m.eval()

        for metric_name in metrics:
            if metric_name.lower() == 'miou':
                self.compute_miou(
                    threshold_output=threshold_output,
                    save_images=save_images,
                )
            elif metric_name.lower() == 'pq':
                self.compute_pq(
                    num_inference_steps=num_inference_steps,
                    threshold_output=threshold_output,
                    save_images=save_images,
                    seed=seed,
                    max_iter=max_iter,
                    dataloader=dataloader,
                    save_model=set_save_model,
                )
        
            else:
                raise NotImplementedError(f'Unknown metric {metric_name}')

            dist.barrier()

        # set models back to train
        for m in models_to_eval:
            m.train()

        return

    
    @torch.no_grad()
    def sample(
        self,
        prompts: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        rgb_latents: Optional[torch.Tensor] = None,
        return_all_latents: bool = False,
        disable_progress_bar: bool = False,
        rgb_images: Optional[torch.Tensor] = None,
        scheduler: Optional[Callable] = None,
        repeat_noise: Optional[bool] = None,
    ) -> torch.Tensor:

        """
        Sample images from the model

        args:
            prompts (list): list of prompts
            num_inference_steps (int): number of inference steps
            guidance_scale (float): guidance scale
            seed (int): seed for the random number generator
            rgb_latents (torch.Tensor): rgb latents to use
            return_all_latents (bool): whether to return all latents
            disable_progress_bar (bool): whether to disable the progress bar
            rgb_images (torch.Tensor): rgb images to use
            scheduler (Callable): scheduler to use

        returns:
            latents (torch.Tensor): latents
        """

        if scheduler is None:
            scheduler = self.noise_scheduler
            scheduler.set_timesteps_inference(num_inference_steps)

        if repeat_noise is None:
            repeat_noise = False

        batch_size = len(prompts)

        # make sure we set the seed for the random number generator locally!
        rng_generator = torch.Generator().manual_seed(seed) if seed is not None else None
        latent_size = rgb_latents.shape[-1] if rgb_latents is not None else self.latent_size
        latents = torch.randn((batch_size, 4, latent_size, latent_size),
                              generator=rng_generator)
        latents = latents.to(device=self.args['gpu'])

        if repeat_noise:
            latents = latents[0:1].repeat(batch_size, 1, 1, 1)
            original_noise = latents.clone()

        multiplier = 1
        encoder_hidden_states = None
        if self.image_descriptor_model is not None:
            assert rgb_images is not None
            inputs_image_descriptors = self.norm_resize_images(rgb_images)
            descriptors = self.image_descriptor_model(inputs_image_descriptors.to(self.weight_dtype))['last_feat']
            descriptors = descriptors.view(descriptors.shape[0], descriptors.shape[1], -1).permute(0, 2, 1)
            encoder_hidden_states = torch.cat([descriptors] * 2)
            encoder_hidden_states = encoder_hidden_states.to(torch.float)
            multiplier = 2
        if self.textencoder is not None:
            text_input = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt")
            text_embeddings = self.textencoder(text_input.input_ids.to(device=self.args['gpu']))[0]
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.textencoder(uncond_input.input_ids.to(device=self.args['gpu']))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            encoder_hidden_states = text_embeddings.to(torch.float)
            multiplier = 2

        latents = latents * scheduler.init_noise_sigma
        if rgb_latents is not None:
            rgb_latents = torch.cat([rgb_latents] * multiplier)

        all_latents = []
        condition = torch.zeros_like(rgb_latents)
        for sample_idx, t in tqdm(enumerate(scheduler.timesteps), disable=disable_progress_bar):
            latent_model_input = torch.cat([latents] * multiplier)

            # predict the noise residual
            if rgb_latents is not None:
                if self.self_condition:
                    inputs = torch.cat([latent_model_input, rgb_latents, condition], dim=1)
                else:
                    inputs = torch.cat([latent_model_input, rgb_latents], dim=1)
            else:
                inputs = latent_model_input
            inputs = inputs.to(self.unet_dtype)

            with torch.autocast('cuda', enabled=self.fp16_scaler is not None):
                noise_pred = self.unet_model(inputs, t, encoder_hidden_states=encoder_hidden_states).sample

            # perform guidance
            if multiplier > 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # update condition
            if self.self_condition:
                condition = scheduler.step(noise_pred, t, latents).pred_original_sample

            # keep = scheduler.step(noise_pred, t, latents).pred_original_sample

            if sample_idx == len(scheduler.timesteps) - 1:
                # compute denoised sample at x_0
                latents = scheduler.step(noise_pred, t, latents).pred_original_sample
            else:
                # compute the previous noisy sample at x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # save the latents
            if return_all_latents:
                all_latents.append(latents)

        # rescale the latents to the original scale
        if return_all_latents:
            return torch.cat(all_latents, dim=0)
        if repeat_noise:
            return latents, original_noise
        return latents

    def crop_padding(self, prediction: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # TODO handle this in a nicer way by loading the coordinates in self.dl
        padding_co = padding_mask.nonzero()
        y_min, y_max = padding_co[:, 0].min(), padding_co[:, 0].max()
        x_min, x_max = padding_co[:, 1].min(), padding_co[:, 1].max()
        prediction = prediction[:, y_min:y_max + 1, x_min:x_max + 1]
        return prediction

    @torch.no_grad()
    def compute_pq(
        self,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        threshold_output: bool = True,
        save_images: bool = False,
        max_iter: Optional[int] = None,
        dataloader: Optional[DataLoader] = None,
        threshold_mode: str = 'max',
        save_model: bool = False,
    ):
        """
        Computes the panoptic quality metric on the validation set
        Currently only class agnostic evaluation is supported
        """

        from ldmseg.evaluations import PanopticEvaluatorAgnostic

        print('Computing PQ metric ...')

        if dataloader is None:
            dataloader = self.dl_val

        meta_data = dataloader.dataset.meta_data
        evaluator = PanopticEvaluatorAgnostic(meta=meta_data)
        evaluator.reset()

        # handle noise scheduler
        scheduler = self.noise_scheduler
        scheduler.set_timesteps_inference(num_inference_steps=num_inference_steps)
        scheduler.move_timesteps_to(self.args['gpu'])
        print(f'Setting noise schedule for eval mode with timesteps {scheduler.timesteps.tolist()} ... ')

        if max_iter is not None:
            print('Running PQ eval on subset percentage {}/{} ...'.format(max_iter, len(dataloader)))

        for batch_idx, data in tqdm(enumerate(dataloader)):
            file_names = [x["image_file"] for x in data['meta']]
            image_ids = [x["image_id"] for x in data['meta']]
            h, w = [x["im_size"][0] for x in data['meta']], [x["im_size"][1] for x in data['meta']]
            rgb_images = data['image'].to(self.args['gpu'], non_blocking=True)
            padding_masks = data['mask'].cuda(self.args['gpu'], non_blocking=True)
            text = data['text']

            rgb_latents, _ = self.encode_inputs(
                rgb_images,
                encode_func=self.vae_image.encode,
                scaling_factor=self.vae_image.scaling_factor,
                resize=self.rgb_size,
            )
            semseg_latents = self.sample(
                text,
                num_inference_steps,
                guidance_scale,
                seed,
                rgb_latents=rgb_latents,
                return_all_latents=False,
                rgb_images=rgb_images,
                scheduler=scheduler,
                disable_progress_bar=True,
            )
            masks_logits = self.decode_latents(
                semseg_latents,
                return_logits=True,
                threshold_output=False,
                rgb_latents=rgb_latents,
                weight_dtype=torch.float32,
            )

            # upsample masks to input size
            masks_logits = F.interpolate(
                masks_logits,
                size=(rgb_images.shape[-2], rgb_images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            # postprocess masks
            processed_results = []
            for image_idx, mask_pred_result in enumerate(masks_logits):

                # crop mask to get rid of padding
                mask_pred_result = self.crop_padding(mask_pred_result, padding_masks[image_idx])

                # interpolate to original size
                mask_pred_result = F.interpolate(
                    mask_pred_result[None, ...].float(),  # [1, C, H, W]
                    size=(h[image_idx], w[image_idx]),
                    mode="bilinear",
                    align_corners=False
                )[0]  # [C, H, W]

                # get panoptic prediction
                panoptic_pred = torch.argmax(mask_pred_result, dim=0)
                if threshold_output:
                    probs = F.softmax(mask_pred_result, dim=0)
                    topk = torch.topk(probs, k=2, dim=0)
                    if threshold_mode == 'topk_diff':
                        topk = torch.topk(probs, k=2, dim=0)
                        probs = topk.values[0] - topk.values[1]
                    else:
                        probs = probs.max(dim=0)[0]
                    panoptic_pred[probs < self.mask_th] = -1

                # move to cpu (to save gpu memory during training)
                panoptic_pred = panoptic_pred.cpu().numpy()
                mask_pred_result = F.sigmoid(mask_pred_result)
                mask_pred_result = mask_pred_result.cpu().numpy()

                processed_results.append({})
                segments_info = []
                for panoptic_label, count_i in zip(*np.unique(panoptic_pred, return_counts=True)):

                    # set small segments to void label (later we add 1 to get 0 for void class)
                    if count_i < self.count_th or panoptic_label in {-1, dataloader.dataset.ignore_label}:
                        panoptic_pred[panoptic_pred == panoptic_label] = -1
                        continue

                    # (optional) also enforce overlap between argmax and thresholded mask
                    original_mask = mask_pred_result[panoptic_label] >= self.mask_th
                    if (panoptic_pred == panoptic_label).sum() / original_mask.sum() < self.overlap_th:
                        panoptic_pred[panoptic_pred == panoptic_label] = -1
                        continue

                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": 1,
                            "isthing": True,
                        }
                    )
                processed_results[-1]["panoptic_seg"] = panoptic_pred + 1, segments_info

            evaluator.process(file_names, image_ids, processed_results)

            if is_main_process() and save_images and batch_idx == 0:
                image_overlayed = self.overlay_predictions(
                    file_names=file_names,
                    processed_results=processed_results,
                    meta_data=meta_data,
                )
                self.log_images_val(
                    latents=semseg_latents,
                    rgb_images=rgb_images,
                    gt_images=data['semseg'],
                    rgb_latents=rgb_latents,
                    inpainting_masks=None,
                    additional_images=[image_overlayed],
                )

            if max_iter is not None and batch_idx > max_iter:
                break

        results = evaluator.evaluate()

        if is_main_process() and save_model:
            if results["panoptic_seg"]["PQ"] > self.best_pq:
                self.best_pq = results["panoptic_seg"]["PQ"]
                print(f'Saving best model with PQ of {self.best_pq} ...')
                epoch_id = self.epoch if hasattr(self, 'epoch') else None
                data = self.construct_save_dict(epoch=epoch_id)
                data['PQ'] = self.best_pq
                torch.save(data, str(self.results_folder / 'best_model.pt'))

        return

    @torch.no_grad()
    def log_images_val(
        self,
        latents: torch.Tensor,
        rgb_images: torch.Tensor,
        gt_images: torch.Tensor,
        rgb_latents: Optional[torch.Tensor] = None,
        inpainting_masks: Optional[torch.Tensor] = None,
        additional_images: Optional[List[np.ndarray]] = None,
        identifier: str = '',
    ) -> None:

        """ Write example predictions to disk
        """

        images = self.decode_latents(latents, rgb_latents=rgb_latents, weight_dtype=torch.float32)
        rgb_images = (255 * rgb_images).cpu().numpy().transpose(0, 2, 3, 1)
        gt_images = self.encode_seg(gt_images.cpu().numpy()).astype(np.uint8)

        # TODO clean this up by adding it to additonal images list
        if inpainting_masks is not None:
            inpainting_masks = F.interpolate(inpainting_masks.float()[:, None],
                                             size=(self.image_size, self.image_size),
                                             mode='nearest')
            inpainting_masks = (255 * inpainting_masks.repeat(1, 3, 1, 1))
            inpainting_masks = inpainting_masks.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            add_inpainting = True
        else:
            inpainting_masks = np.zeros_like(gt_images)
            add_inpainting = False

        nimgs = self.batch_size_val
        size = self.image_size
        offset = int(0.02 * size)
        rgb_array = np.zeros((size, nimgs * (size + offset), 3), dtype=np.uint8)
        gt_array = np.zeros((size, nimgs * (size + offset), 3), dtype=np.uint8)
        gen_array = np.zeros((size, nimgs * (size + offset), 3), dtype=np.uint8)
        inpaint_array = np.zeros((size, nimgs * (size + offset), 3), dtype=np.uint8)
        ptr = 0
        for idx, (rgb, gt, gen_image, inpaint_image) in enumerate(zip(
            rgb_images[:nimgs],
            gt_images[:nimgs],
            images[:nimgs],
            inpainting_masks[:nimgs],
        )):

            rgb_array[:, ptr:ptr + size, :] = rgb
            gt_array[:, ptr:ptr + size, :] = gt
            gen_array[:, ptr:ptr + size, :] = gen_image
            inpaint_array[:, ptr:ptr + size, :] = inpaint_image
            ptr += size + offset

        stacked_images = [rgb_array, gt_array, gen_array]
        if add_inpainting:
            stacked_images.append(inpaint_array)
        if additional_images is not None:
            stacked_images.extend(additional_images)
        self.write_images(np.vstack(stacked_images), f'overview{identifier}.png')

        return

    @torch.no_grad()
    def log_images_train(
        self,
        log_dict: dict,
        *,
        noisy_latents: torch.Tensor,
        pred_latents: torch.Tensor,
        rgb_latents: torch.Tensor,
        rgb_images: torch.Tensor,
        gt_images: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        original_latents: Optional[torch.Tensor] = None,
        noisy_prompts: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        inpainting_masks: Optional[torch.Tensor] = None,
        scheduler: Optional[Callable] = None,
    ) -> None:

        """ Log images to wandb or save them to disk
        """

        self.unet_model.eval()

        if self.decoder is not None:
            self.decoder.eval()

        latents = self.sample(
            noisy_prompts, num_inference_steps,
            guidance_scale, seed,
            rgb_latents=rgb_latents, rgb_images=rgb_images,
            scheduler=scheduler,
        )

        images = self.decode_latents(latents, rgb_latents=rgb_latents, weight_dtype=torch.float32)

        if self.use_wandb:
            timesteps = timesteps.cpu().numpy()
            if 'images' not in log_dict:
                log_dict['images'] = {}
            # noisy images
            noisy_images = self.decode_latents(noisy_latents, rgb_latents=rgb_latents, weight_dtype=torch.float32)
            pred_images = self.decode_latents(pred_latents, rgb_latents=rgb_latents, weight_dtype=torch.float32)
            rgb_images = (255*rgb_images).cpu().numpy().transpose(0, 2, 3, 1)

            log_dict['images'].update({
                'rgb_to_semseg':
                [
                    wandb.Image(image, caption=caption + '_img')
                    for image, caption in zip(images, noisy_prompts)
                ],
                'noisy_semseg':
                [
                    wandb.Image(image, caption=caption + '_' + str(t) + '_noisy')
                    for image, caption, t in zip(noisy_images, noisy_prompts, timesteps)
                ],
                'predicted_from_noisy_semseg':
                [
                    wandb.Image(image, caption=caption + '_' + str(t) + '_pred')
                    for image, caption, t in zip(pred_images, noisy_prompts, timesteps)
                ],
                'rgb_images':
                [
                    wandb.Image(image, caption=caption + '_' + str(t) + '_rgb')
                    for image, caption, t in zip(rgb_images, noisy_prompts, timesteps)
                ],
            }
            )

        else:
            nimgs = self.batch_size_val

            pred_images = self.decode_latents(pred_latents, rgb_latents=rgb_latents)
            noisy_images = self.decode_latents(noisy_latents, rgb_latents=rgb_latents)
            sanity_images = self.decode_latents(original_latents, rgb_latents=rgb_latents)
            rgb_images = (255 * rgb_images).cpu().numpy().transpose(0, 2, 3, 1)
            gt_images = self.encode_seg(gt_images.cpu().numpy()).astype(np.uint8)
            if inpainting_masks is not None:
                inpainting_masks = F.interpolate(inpainting_masks.float()[:, None],
                                                 size=(self.image_size, self.image_size),
                                                 mode='nearest')
                inpainting_masks = (255 * inpainting_masks.repeat(1, 3, 1, 1))
                inpainting_masks = inpainting_masks.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                add_inpainting = True
            else:
                inpainting_masks = np.zeros_like(gt_images)
                add_inpainting = False

            ptr = 0
            size = self.image_size
            offset = int(0.02 * size)
            pred_array = np.zeros((size, nimgs * (size + offset), 3), dtype=np.uint8)
            rgb_array = np.zeros((size, nimgs * (size + offset), 3), dtype=np.uint8)
            gt_array = np.zeros((size, nimgs * (size + offset), 3), dtype=np.uint8)
            noisy_array = np.zeros((size, nimgs * (size + offset), 3), dtype=np.uint8)
            sanity_array = np.zeros((size, nimgs * (size + offset), 3), dtype=np.uint8)
            gen_array = np.zeros((size, nimgs * (size + offset), 3), dtype=np.uint8)
            inpaint_array = np.zeros((size, nimgs * (size + offset), 3), dtype=np.uint8)
            for idx, (image, rgb, gt, noisy_image, sanity_image, gen_image, inpaint_image) in enumerate(zip(
                pred_images[:nimgs],
                rgb_images[:nimgs],
                gt_images[:nimgs],
                noisy_images[:nimgs],
                sanity_images[:nimgs],
                images[:nimgs],
                inpainting_masks[:nimgs],
            )):

                pred_array[:, ptr:ptr + size, :] = image
                rgb_array[:, ptr:ptr + size, :] = rgb
                gt_array[:, ptr:ptr + size, :] = gt
                noisy_array[:, ptr:ptr + size, :] = noisy_image
                sanity_array[:, ptr:ptr + size, :] = sanity_image
                gen_array[:, ptr:ptr + size, :] = gen_image
                inpaint_array[:, ptr:ptr + size, :] = inpaint_image
                ptr += size + offset

            stacked_images = [rgb_array, gt_array, sanity_array, noisy_array, pred_array, gen_array]
            if add_inpainting:
                stacked_images.append(inpaint_array)
            self.write_images(np.vstack(stacked_images), 'all.png')

            print(f'saved predictions during train with timesteps {timesteps.cpu().tolist()}')

        if self.decoder is not None:
            self.decoder.eval()
        else:
            self.unet_model.train()
        return

    def overlay_predictions(self, file_names: List[str], processed_results: dict,
                            meta_data: dict, identifier: str = ''):
        from ldmseg.utils import MyVisualizer
        import cv2

        bs = len(file_names)
        size = self.image_size
        offset = int(0.02 * size)
        panoptic_overlay_array = np.zeros((size, bs * (size + offset), 3), dtype=np.uint8)

        ptr = 0
        for file_name, processed_res in zip(file_names, processed_results):
            curr_image = Image.open(file_name).convert("RGB")
            curr_image = np.asarray(curr_image)
            vis_image = MyVisualizer(curr_image, metadata=meta_data, scale=1.0)
            res = vis_image.draw_panoptic_seg(
                torch.from_numpy(processed_res["panoptic_seg"][0]),
                processed_res["panoptic_seg"][1],
                suppress_thing_labels=True,
                random_colors=True,
                alpha=0.8,
            )
            ratio = size / max(curr_image.shape[:2])
            h_new, w_new = int(curr_image.shape[0] * ratio), int(curr_image.shape[1] * ratio)
            image_overlay = cv2.resize(res.get_image(), (w_new, h_new), interpolation=cv2.INTER_CUBIC)
            panoptic_overlay_array[:h_new, ptr:ptr+w_new] = image_overlay
            ptr += size + offset

        self.write_images(panoptic_overlay_array, f'panoptic_overlay{identifier}.jpg')
        return panoptic_overlay_array

    @torch.no_grad()
    def visualize_noise_schedule(self, seed: Optional[int] = None):
        """ Visualize the noise schedule
        """
        if not is_main_process():
            return
        print("Visualizing noise schedule ...")

        batch = next(iter(self.dl))
        model_inputs = self.process_inputs(batch)
        # random_idx = np.random.randint(0, self.batch_size)
        random_idx = 0
        seed = 42
        latents = model_inputs.latents[random_idx:random_idx+1]
        rgb_image = model_inputs.rgb_images[random_idx:random_idx+1]
        rgb_image = (255 * rgb_image).cpu().numpy().transpose(0, 2, 3, 1)
        scheduler = self.noise_scheduler

        rng_generator = torch.Generator().manual_seed(seed) if seed is not None else None
        noise = torch.randn(latents.shape, generator=rng_generator)
        noise = noise.to(latents.device)
        ptr, size, step = 0, 512, 125
        offset = int(0.02 * size)
        array = np.full((size,  (scheduler.num_train_timesteps // step + 1) * (size + offset), 3),
                        dtype=np.uint8, fill_value=255)

        array[:, ptr:ptr+size, :] = rgb_image[0]
        ptr += size + offset
        for t in range(0, scheduler.num_train_timesteps, step):
            timesteps = torch.tensor(t, device=latents.device, dtype=torch.long).unsqueeze(0)
            noisy_latent = self.noise_scheduler.add_noise(latents, noise, timesteps)
            array[:, ptr:ptr + size, :] = self.decode_latents(noisy_latent)[0]
            ptr += size + offset
        self.write_images(array, 'noise_schedule.jpg')

    @torch.no_grad()
    def visualize_noise(self, num_inference_steps: int = 50, guidance_scale: float = 7.5,):
        """ Visualize impact of noise
        """
        if not is_main_process():
            return
        print("Visualizing impact of noise queries ...")

        # handle noise scheduler
        scheduler = self.noise_scheduler
        scheduler.set_timesteps_inference(num_inference_steps=self.num_inference_steps)
        scheduler.move_timesteps_to(self.args['gpu'])
        print(f'Setting noise schedule for eval mode with timesteps {scheduler.timesteps.tolist()} ... ')

        self.unet_model.eval()
        if self.decoder is not None:
            self.decoder.eval()

        nrows = 4
        results = []
        data = next(iter(self.dl))
        rgb_images = data['image'].to(self.args['gpu'], non_blocking=True)
        text = data['text']
        bs, _, H, W = rgb_images.shape
        seed = 1
        rgbs = (255 * rgb_images).cpu().numpy().transpose(0, 2, 3, 1)
        ptr, size = 0, H
        offset = int(0.02 * size)
        array = np.full((size,  (bs + 1) * (size + offset), 3),
                        dtype=np.uint8, fill_value=255)
        ptr += size + offset
        for rgb in rgbs:
            array[:, ptr:ptr + size, :] = rgb
            ptr += size + offset
        row_zeros = np.full((offset, (bs + 1) * (size + offset), 3), dtype=np.uint8, fill_value=255)
        results.append(array)
        results.append(row_zeros)

        # get latents
        rgb_latents, _ = self.encode_inputs(
            rgb_images,
            encode_func=self.vae_image.encode,
            scaling_factor=self.vae_image.scaling_factor,
            resize=self.rgb_size,
        )

        for row in tqdm(range(nrows)):
            semseg_latents, noise_map = self.sample(
                text,
                num_inference_steps,
                guidance_scale,
                seed=seed + row,
                rgb_latents=rgb_latents,
                return_all_latents=False,
                rgb_images=rgb_images,
                scheduler=scheduler,
                disable_progress_bar=True,
                repeat_noise=True,
            )
            masks_logits = self.decode_latents(
                semseg_latents,
                return_logits=True,
                threshold_output=False,
                rgb_latents=rgb_latents,
                weight_dtype=torch.float32,
            )

            # upsample noise map
            noise_logits = self.decode_latents(
                noise_map,
                return_logits=True,
                threshold_output=False,
                rgb_latents=rgb_latents,
                weight_dtype=torch.float32,
            )
            masks_logits = F.interpolate(
                masks_logits,
                size=(rgb_images.shape[-2], rgb_images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            noise = torch.argmax(noise_logits, dim=1)
            noise = noise.cpu().numpy()
            noise = self.encode_seg(noise).astype(np.uint8)

            # upsample masks to input size
            masks_logits = F.interpolate(
                masks_logits,
                size=(rgb_images.shape[-2], rgb_images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            predictions = torch.argmax(masks_logits, dim=1)
            probs = F.softmax(masks_logits, dim=1)
            probs = probs.max(dim=1)[0]
            predictions[probs < self.mask_th] = self.ds.ignore_label
            predictions = predictions.cpu().numpy()
            predictions = self.encode_seg(predictions).astype(np.uint8)

            ptr = 0
            array = np.full((size,  (bs + 1) * (size + offset), 3),
                            dtype=np.uint8, fill_value=255)
            array[:, ptr:ptr + size, :] = noise[0]
            ptr += size + offset
            for pred in predictions:
                array[:, ptr:ptr + size, :] = pred
                ptr += size + offset
            results.append(array)
            if row < nrows - 1:
                results.append(row_zeros)
        results = np.vstack(results)
        self.write_images(results, 'noise_queries.jpg')

    def _reshape_images(self, images: np.ndarray) -> np.ndarray:
        """ Reshape images for visualization
        """
        bs, h, w, c = images.shape
        images = images.transpose(1, 0, 2, 3).astype(np.uint8)
        images = images.reshape(h, bs * w, c)
        return images

    def write_images(self, images: Union[np.ndarray, List[np.ndarray]], path_names: Union[str, List[str]]) -> None:
        """ Write images to the visualization directory
        """
        if isinstance(images, np.ndarray):
            images = [images]
        if isinstance(path_names, str):
            path_names = [path_names]
        for image, file_name in zip(images, path_names):
            image = Image.fromarray(image)
            image.save(os.path.join(self.visualization_dir, file_name))

    def get_lr_scheduler(
        self,
        name: str = 'cosine',
        final_lr: float = 0.0,
        warmup_iters: Optional[int] = None,
        lr_scaling: bool = False,
    ) -> Optional[np.ndarray]:
        """ Returns the learning rate scheduler
        """

        # (optional) lr scaling
        if lr_scaling:
            self.eff_lr = self.lr * (self.eff_batch_size * get_world_size()) / 64.  # linear scaling rule
        else:
            self.eff_lr = self.lr

        if name == 'cosine':
            lr_schedule = cosine_scheduler(
                self.eff_lr,
                final_lr,
                self.epochs,
                self.num_iters_per_epoch,
                warmup_iters=warmup_iters,
            )
        elif name == 'warmup':
            lr_schedule = warmup_scheduler(
                self.eff_lr,
                final_lr,
                self.epochs,
                self.num_iters_per_epoch,
                warmup_iters=warmup_iters,
            )
        elif name == 'step':
            lr_schedule = step_scheduler(
                self.eff_lr,
                final_lr,
                self.epochs,
                self.num_iters_per_epoch,
                decay_epochs=[self.epochs // 2, 3 * self.epochs // 4],
                decay_rate=0.1,
                warmup_iters=warmup_iters,
            )
        else:
            raise NotImplementedError(f'Unknown lr scheduler: {name}')

        print(colored(
            f'Using lr scheduler {name} with '
            f'effective lr: {self.eff_lr:.3e}, '
            f'final lr: {final_lr:.3e}, '
            f'warmup iters {warmup_iters}',
            'yellow'))
        return lr_schedule

    def construct_save_dict(self, epoch: Optional[int] = None) -> dict:
        if isinstance(self.unet_model, nn.parallel.DistributedDataParallel):
            unet_state = self.unet_model.module.state_dict()
        else:
            unet_state = self.unet_model.state_dict()

        # consolidate optimizer state_dict on rank 0
        if self.save_optim and isinstance(self.opt, dist.optim.ZeroRedundancyOptimizer):
            self.opt.consolidate_state_dict(0)
            print('Consolidated optimizer state dict on RANK 0')

        data = {
            'step': self.step,
            'epoch': epoch,
            'vae_image': self.vae_image.state_dict(),
            'vae_semseg': self.vae_semseg.state_dict(),
            'unet': unet_state,
            'ema': self.ema_unet_model.state_dict() if self.use_ema else None,
            'opt': self.opt.state_dict() if self.save_optim else None,
            'p': self.p,
            'scaler': self.fp16_scaler.state_dict() if self.fp16_scaler is not None else None
        }

        return data

    def save(self, epoch: int) -> None:
        """ Saves the model
        """

        if not is_main_process():
            return

        data = self.construct_save_dict(epoch)
        torch.save(data, str(self.results_folder / 'model.pt'))

    def resume(self, load_vae: bool = True) -> None:
        """ Resumes training from a saved model
        """

        model_path = str(self.results_folder / 'model.pt')
        if not os.path.exists(model_path):
            print(colored(f'No saved model at {model_path} ...', 'blue'))
            return

        # load model
        print(colored(f'Resuming model from {model_path} ...', 'blue'))
        data = torch.load(model_path, map_location='cpu')
        if isinstance(self.unet_model, nn.parallel.DistributedDataParallel):
            self.unet_model.module.load_state_dict(data['unet'])
        else:
            self.unet_model.load_state_dict(data['unet'])

        if load_vae:
            print(colored('Loading VAE ...', 'blue'))
            self.vae_image.load_state_dict(data['vae_image'])
            self.vae_semseg.load_state_dict(data['vae_semseg'])

        self.start_epoch = data['epoch'] + 1  # start counting from 0
        self.step = (data['epoch'] + 1) * self.num_iters_per_epoch + 1
        if data['opt'] is not None:
            self.opt.load_state_dict(data['opt'])
        if self.use_ema and is_main_process():
            self.ema_unet_model.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if data['scaler'] is not None:
            self.fp16_scaler.load_state_dict(data['scaler'])

        return data

    def load(self, load_vae: bool = True, model_path: Optional[str] = None) -> None:
        """ Resumes training from a saved model
        """

        print(colored('Start loading function ...', 'blue'))
        if model_path is None:
            print('No loading model path specified ...')
            return

        if not os.path.exists(model_path):
            print(colored(f'No saved model at {model_path} ...', 'blue'))
            return

        print(colored(f'Loading saved model on all gpus from {model_path} ...', 'blue'))
        data = torch.load(model_path, map_location='cpu')

        if isinstance(self.unet_model, nn.parallel.DistributedDataParallel):
            self.unet_model.module.load_state_dict(data['unet'])
        else:
            self.unet_model.load_state_dict(data['unet'])

        if load_vae:
            print(colored('Loading VAE ...', 'blue'))
            # self.vae_image.load_state_dict(data['vae_image'])
            self.vae_semseg.load_state_dict(data['vae_semseg'])
        if self.use_ema:
            self.ema_unet_model.load_state_dict(data['ema'])

        return data
