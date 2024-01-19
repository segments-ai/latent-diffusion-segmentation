"""
Author: Wouter Van Gansbeke

This file contains the noise scheduler for the diffusion process.
Based on the implement. in the diffusers library (Apache License): https://https://github.com/huggingface/diffusers
Added features to DDIM scheduler(https://arxiv.org/abs/2102.09672), in summary:
- Define method to remove noise from the noisy samples according to the adopted scheduler.
- Define loss weights for each timestep. The weights are used to scale the loss for each timestep.
  (i.e., small timesteps are weighted less than large timesteps.)
- Add glide cosine schedule from diffusers to DDIM as well.
- Use a `step_offset` by default during inference for sampling segmentation maps from Guassian noise.
"""

import math
import torch
import numpy as np
from ldmseg.utils import OutputDict
from typing import Optional, Union


class DDIMNoiseSchedulerOutput(OutputDict):
    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class DDIMNoiseScheduler(object):
    """
    Noise scheduler for the diffusion process.
    Implementation is adapted from the diffusers library
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        weight: str = 'none',
        max_snr: float = 5.0,
        device: Union[str, torch.device] = None,
        verbose: bool = True,
    ):
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = self.get_betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # compute loss weights
        self.compute_loss_weights(mode=weight, max_snr=max_snr)
        self.weights = self.weights.to(device)

        # set other parameters
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.steps_offset = steps_offset
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.init_noise_sigma = 1.0
        self.verbose = verbose

    def compute_loss_weights(self, mode='max_clamp_snr', max_snr=5.0):
        """
        Compute loss weights for each timestep. The weights are used to scale the loss of each timestep.
        Small timesteps are weighted less than large timesteps.
        """

        assert mode in ['inverse_log_snr', 'max_clamp_snr', 'linear', 'fixed', 'none']
        self.weight_mode = mode
        snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        if mode == 'inverse_log_snr':
            self.weights = torch.log(1. / snr).clamp(min=1)
            self.weights /= self.weights[-1]  # normalize
        elif mode == 'max_clamp_snr':
            self.weights = snr.clamp(max=max_snr) / snr
        elif mode == 'fixed':
            self.weights = snr
            self.weights[:len(self.weights) // 4] = 0.1
        elif mode == 'linear':
            self.weights = torch.arange(1, len(snr) + 1) / len(snr)
        else:
            self.weights = torch.ones_like(snr)

    def set_timesteps_inference(self, num_inference_steps: int, device: Union[str, torch.device] = None, tmin: int = 0):
        """
        Set the timesteps for inference. This is used to compute the noise schedule for inference.
        We shift the timesteps by `steps_offset` to make sure the final timestep is always included (i.e., t = 999)
        """

        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        self.steps_offset = step_ratio - 1
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.steps_offset
        self.timesteps = self.timesteps[self.timesteps >= tmin]

    def move_timesteps_to(self, device: Union[str, torch.device]):
        """ Move timesteps to `device`
        """
        self.timesteps = self.timesteps.to(device)

    def get_betas_for_alpha_bar(self, num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
        """
        Used for Glide cosine schedule.
        Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
        (1-beta) over time from t = [0,1].
        """

        def alpha_bar(time_step):
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
        scale: float = 1.0,
        mask_noise_perc: Optional[float] = None,
    ) -> torch.FloatTensor:
        """
        Add noise to the original samples according to the noise schedule.
        The core function of the diffusion process.
        """

        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        if mask_noise_perc is not None:
            # fill percentage of the mask with zeros (i.e. remove noise)
            mask = torch.rand_like(original_samples) < mask_noise_perc
            noise *= mask

        noisy_samples = sqrt_alpha_prod * scale * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    @torch.no_grad()
    def remove_noise(
        self,
        noisy_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        """
        Remove predicted noise from the noisy samples according to the defined noise scheduler.
        """

        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=noisy_samples.device, dtype=noisy_samples.dtype)
        timesteps = timesteps.to(noisy_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        original_samples = (noisy_samples - sqrt_one_minus_alpha_prod * noise) / (sqrt_alpha_prod * scale)
        return original_samples

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        use_clipped_model_output: bool = False,
    ) -> DDIMNoiseSchedulerOutput:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        """

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise NotImplementedError

        # 4. Clip or threshold "predicted x_0"
        if self.thresholding:
            raise NotImplementedError
        elif self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t"
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise"
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return DDIMNoiseSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def __str__(self) -> str:
        print_weights = self.weights if self.verbose else 'VerboseDisabled'
        return f"DDIMScheduler(num_inference_steps={self.num_inference_steps}, " \
               f"num_train_timesteps={self.num_train_timesteps}, " \
               f"prediction_type={self.prediction_type}, " \
               f"beta_start={self.beta_start}, " \
               f"beta_end={self.beta_end}, " \
               f"beta_schedule={self.beta_schedule}, " \
               f"clip_sample={self.clip_sample}, " \
               f"clip_sample_range={self.clip_sample_range}, " \
               f"thresholding={self.thresholding}, " \
               f"dynamic_thresholding_ratio={self.dynamic_thresholding_ratio}, " \
               f"steps_offset={self.steps_offset}, " \
               f"weight_mode={self.weight_mode}, " \
               f"weights={print_weights})"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.num_train_timesteps
