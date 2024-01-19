"""
Author: Wouter Van Gansbeke

Main file for training auto-encoders and vaes
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import sys
import json
import hydra
import wandb
import builtins
from termcolor import colored
from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from diffusers import AutoencoderKL

from ldmseg.models import GeneralVAESeg
from ldmseg.trainers import TrainerAE
from ldmseg.utils import prepare_config, Logger, is_main_process


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function that calls multiple workers for distributed training

    args:
        cfg (DictConfig): configuration file as a dictionary
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

    if cfg_dist['dist_url'] == "env://" and cfg_dist['world_size'] == -1:
        cfg_dist['world_size'] = int(os.environ["WORLD_SIZE"])
    cfg_dist['distributed'] = cfg_dist['world_size'] > 1 or cfg_dist['multiprocessing_distributed']
    # handle debug mode
    if cfg['debug']:
        print(colored("Running in debug mode!", "red"))
        cfg_dist['world_size'] = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cfg_dataset['train_kwargs']['num_workers'] = 0
        cfg_dataset['eval_kwargs']['num_workers'] = 0
    ngpus_per_node = torch.cuda.device_count()
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    cfg_dist['world_size'] = ngpus_per_node * cfg_dist['world_size']
    print(colored(f"World size: {cfg_dist['world_size']}", 'blue'))
    # main_worker process function
    if cfg['debug']:
        main_worker(0, ngpus_per_node, cfg_dist, cfg_dataset, project_name)
    else:
        # Use torch.multiprocessing.spawn to launch distributed processes
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg_dist, cfg_dataset, project_name))


def main_worker(
    gpu: int,
    ngpus_per_node: int,
    cfg_dist: Dict[str, Any],
    p: Dict[str, Any],
    name: str = 'segmentation_diffusion'
) -> None:
    """
    A single worker for distributed training

    args:
        gpu (int): local index of the gpu per node
        ngpus_per_node (int): number of gpus per node
        cfg_dist (Dict): arguments for distributed training
        p (Dict): arguments for training
        name (str): name of the run
    """

    cfg_dist['gpu'] = gpu
    cfg_dist['ngpus_per_node'] = ngpus_per_node
    if cfg_dist['multiprocessing_distributed'] and cfg_dist['gpu'] != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if cfg_dist['gpu'] is not None:
        print("Use GPU: {} for printing".format(cfg_dist['gpu']))

    if cfg_dist['distributed']:
        if cfg_dist['dist_url'] == "env://" and cfg_dist['rank'] == -1:
            cfg_dist['rank'] = int(os.environ["RANK"])
        if cfg_dist['multiprocessing_distributed']:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg_dist['rank'] = cfg_dist['rank'] * ngpus_per_node + gpu
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

    # get param dicts
    train_params, eval_params = p['train_kwargs'], p['eval_kwargs']

    # define model
    pretrained_model_path = p['pretrained_model_path']
    print('pretrained_model_path', pretrained_model_path)
    vae_encoder = None
    if p['shared_vae_encoder']:
        vae_image = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        vae_encoder = torch.nn.Sequential(vae_image.encoder, vae_image.quant_conv)
    vae = GeneralVAESeg(**p['vae_model_kwargs'], encoder=vae_encoder)
    print(vae)

    # gradient checkpointing
    if p['train_kwargs']['gradient_checkpointing']:
        vae.enable_gradient_checkpointing()

    # load model on gpu
    vae = vae.to(cfg_dist['gpu'])

    # print number of trainable parameters
    print(colored(
        f"Number of trainable parameters: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6 :.2f}M",
        'yellow'))

    if cfg_dist['distributed']:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        assert cfg_dist['gpu'] is not None
        vae = torch.nn.parallel.DistributedDataParallel(
            vae, device_ids=[cfg_dist['gpu']],
            find_unused_parameters=train_params['find_unused_parameters'],
            gradient_as_bucket_view=train_params['gradient_as_bucket_view'],
        )
        print(colored(f"Batch size per process is {train_params['batch_size']}", 'yellow'))
        print(colored(f"Workers per process is {train_params['num_workers']}", 'yellow'))
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define trainer
    trainer = TrainerAE(
        p,
        vae,
        args=cfg_dist,                                         # distributed training args
        results_folder=p['output_dir'],                        # output directory
        save_and_sample_every=eval_params['vis_every'],        # save and sample every n iters
        cudnn_on=train_params['cudnn'],                        # turn on cudnn
        fp16=train_params['fp16'],                             # turn on floating point 16
    )

    # resume training from checkpoint
    trainer.resume()  # looks for model at run_idx (automatically determined)
    if p['load_path'] is not None:
        trainer.load(model_path=p['load_path'])  # looks for model at load_path

    if p['eval_only']:
        trainer.compute_metrics(['miou', 'pq'])
        return

    # train
    trainer.train_loop()

    # terminate wandb
    if p['wandb'] and is_main_process():
        wandb.finish()


if __name__ == "__main__":
    main()
