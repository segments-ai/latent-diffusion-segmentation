"""
Author: Wouter Van Gansbeke

Main file for training diffusion models
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import sys
import json
import wandb
import builtins
import hydra
from typing import Dict, Any
from termcolor import colored
from datetime import datetime
from omegaconf import OmegaConf, DictConfig

import torch
import torch.nn as nn
import torch.distributed as dist

from ldmseg.trainers import TrainerDiffusion
from ldmseg.models import UNet, GeneralVAESeg, GeneralVAEImage, get_image_descriptor_model
from ldmseg.schedulers import DDIMNoiseScheduler
from ldmseg.utils import prepare_config, Logger, is_main_process


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main function that calls multiple workers for distributed training
    """

    # define configs
    cfg = OmegaConf.to_object(cfg)
    wandb.config = cfg
    cfg_dist = cfg['distributed']
    cfg_dataset = cfg['datasets']
    cfg_base = cfg['base']
    project_dir = cfg['setup']
    cfg_dataset = cfg_base | cfg_dataset  # overwrite base configs with dataset specific configs
    root_dir = os.path.join(cfg['env']['root_dir'], project_dir)
    data_dir = cfg['env']['data_dir']
    cfg_dataset, project_name = prepare_config(cfg_dataset, root_dir, data_dir, run_idx=cfg['run_idx'])
    project_name = f"{cfg_dataset['train_db_name']}_{project_name}"
    print(colored(f"Project name: {project_name}", 'red'))
    cfg_dist['distributed'] = True

    ngpus_per_node = torch.cuda.device_count()
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    cfg_dist['rank'] = int(os.environ["SLURM_PROCID"])
    cfg_dist['world_size'] = int(os.environ["SLURM_NNODES"]) * int(
        os.environ["SLURM_TASKS_PER_NODE"][0])
    print(colored(f"World size: {cfg_dist['world_size']}", 'blue'))
    cfg_dist['gpu'] = cfg_dist['rank'] % torch.cuda.device_count()
    print(colored(f"GPU: {cfg_dist['gpu']}", 'blue'))
    cfg_dist['ngpus_per_node'] = ngpus_per_node

    # main_worker process function
    main_worker(cfg_dist, cfg_dataset, project_name)
    return


def main_worker(
    cfg_dist: Dict[str, Any],
    p: Dict[str, Any],
    name: str = 'simple_diffusion'
) -> None:
    """
    A single worker for distributed training

    args:
        cfg_dist (Dict[str, Any]): distributed configs
        p (Dict[str, Any]): training configs
        name (str): name of the experiment
    """

    if cfg_dist['multiprocessing_distributed'] and cfg_dist['gpu'] != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if cfg_dist['gpu'] is not None:
        print("Use GPU: {} for printing".format(cfg_dist['gpu']))

    if cfg_dist['distributed']:
        assert cfg_dist['multiprocessing_distributed']
        # cfg_dist['rank'] = cfg_dist['rank'] * ngpus_per_node + gpu
        print(colored("Starting distributed training", "yellow"))
        dist.init_process_group(backend=cfg_dist['dist_backend'],
                                init_method=cfg_dist['dist_url'],
                                world_size=cfg_dist['world_size'],
                                rank=cfg_dist['rank'])

    print(colored("Initialized distributed training", "yellow"))
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(cfg_dist['gpu'])

    # logging and printing with wandb
    if p['wandb'] and is_main_process():
        wandb.init(name=name, project="ddm")
    sys.stdout = Logger(os.path.join(p['output_dir'], f'log_file_gpu_{cfg_dist["gpu"]}.txt'))

    # print configs
    p['name'] = name
    readable_p = json.dumps(p, indent=4, sort_keys=True)
    print(colored(readable_p, 'red'))
    print(colored(datetime.now(), 'yellow'))

    # get model params
    train_params, eval_params = p['train_kwargs'], p['eval_kwargs']
    pretrained_model_path = p['pretrained_model_path']
    print('Pretrained model path:', pretrained_model_path)

    semseg_vae_encoder = None
    cache_dir = os.path.join(p['data_dir'], 'cache')
    vae_image = GeneralVAEImage.from_pretrained(
        pretrained_model_path, subfolder="vae", cache_dir=cache_dir)
    vae_image.decoder = nn.Identity()  # remove the decoder
    vae_image.set_scaling_factor(p['image_scaling_factor'])
    if p['shared_vae_encoder']:
        semseg_vae_encoder = torch.nn.Sequential(vae_image.encoder, vae_image.quant_conv)  # NOTE: deepcopy is safer
    vae_semseg = GeneralVAESeg(**p['vae_model_kwargs'], encoder=semseg_vae_encoder)
    print(f"Scaling factors: image {vae_image.scaling_factor}, semseg {vae_semseg.scaling_factor}")

    # load the pretrained UNet model
    unet = UNet.from_pretrained(pretrained_model_path, subfolder="unet", cache_dir=cache_dir)

    # gradient checkpointing
    if p['train_kwargs']['gradient_checkpointing']:
        print(colored('Gradient checkpointing is enabled', 'yellow'))
        unet.enable_gradient_checkpointing()

    # load the pretrained image descriptor model
    # leverage prior knowledge by using descriptors: CLIP, DINO, MAE, or learnable
    image_descriptor_model, text_encoder, tokenizer = get_image_descriptor_model(
        train_params['image_descriptors'], pretrained_model_path, unet
    )
    # modify the encoder of the UNet and freeze layers
    unet.modify_encoder(**p['model_kwargs'])
    unet.freeze_layers(p['train_kwargs']['freeze_layers'])
    print('time embedding frozen')

    # define weight dtype
    assert train_params['weight_dtype'] in ['float32', 'float16']
    weight_dtype = torch.float32 if train_params['weight_dtype'] == 'float32' else torch.float16
    vae_image = vae_image.to(cfg_dist['gpu'], dtype=weight_dtype)
    vae_semseg = vae_semseg.to(cfg_dist['gpu'], dtype=torch.float32)
    unet = unet.to(cfg_dist['gpu'], dtype=torch.float32)  # unet is always fp32
    if image_descriptor_model is not None:
        image_descriptor_model = image_descriptor_model.to(cfg_dist['gpu'], dtype=weight_dtype)
    if text_encoder is not None:
        text_encoder = text_encoder.to(cfg_dist['gpu'], dtype=weight_dtype)

    # define noise scheduler
    noise_scheduler = DDIMNoiseScheduler(**p['noise_scheduler_kwargs'], device=cfg_dist['gpu'])
    print(noise_scheduler)

    # print number of trainable parameters
    print(colored(
        f"Number of trainable parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M",
        'yellow'))

    if cfg_dist['distributed']:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        assert cfg_dist['gpu'] is not None
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[cfg_dist['gpu']],
            find_unused_parameters=p['train_kwargs']['find_unused_parameters'],
            gradient_as_bucket_view=p['train_kwargs']['gradient_as_bucket_view'],
        )
        print(colored(f"Batch size per process is {train_params['batch_size']}", 'yellow'))
        print(colored(f"Workers per process is {train_params['num_workers']}", 'yellow'))
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define trainer
    trainer = TrainerDiffusion(
        p,
        vae_image, vae_semseg, unet, tokenizer, text_encoder,  # models
        noise_scheduler,                                       # noise scheduler for diffusion
        image_descriptor_model=image_descriptor_model,         # image descriptor model
        ema_unet=None,                                         # ema model for unet
        args=cfg_dist,                                         # distributed training args
        results_folder=p['output_dir'],                        # output directory
        save_and_sample_every=eval_params['vis_every'],        # save and sample every n iters
        cudnn_on=train_params['cudnn'],                        # turn on cudnn
        fp16=train_params['fp16'],                             # turn on floating point 16
        ema_on=p['ema_on'],                                    # turn on exponential moving average
        weight_dtype=weight_dtype,                             # weight dtype
    )

    # resume training from checkpoint
    trainer.resume(load_vae=True)  # looks for model at run_idx (automatically determined)
    trainer.load(model_path=p['load_path'], load_vae=True)  # looks for model at load_path

    # only evaluate
    if p['eval_only']:
        max_iter = None
        trainer.compute_metrics(
            metrics=['pq'],
            dataloader=trainer.dl_val,
            threshold_output=True,
            save_images=True,
            seed=42,
            max_iter=max_iter,
            models_to_eval=[trainer.unet_model],
            num_inference_steps=trainer.num_inference_steps,
        )
        dist.barrier()
        return

    # train
    trainer.train_loop()

    # terminate wandb
    if p['wandb'] and is_main_process():
        wandb.finish()


if __name__ == "__main__":
    main()
