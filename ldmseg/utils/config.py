"""
Author: Wouter Van Gansbeke

Functions to handle configuration files
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import yaml
import errno
from easydict import EasyDict


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def load_config(config_file_exp):
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    for k, v in config.items():
        cfg[k] = v

    return cfg


def create_config(config_file_env, config_file_exp, run_idx=None):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Num classes
    if cfg['train_db_name'].lower() not in ['coco', 'coco_panoptic', 'cityscapes', 'cityscapes+coco']:
        raise ValueError('Invalid train db name {}'.format(cfg['train_db_name']))

    # Paths
    subdir = os.path.join(root_dir, config_file_exp.split('/')[-2])
    mkdir_if_missing(subdir)
    output_dir = os.path.join(subdir, os.path.basename(config_file_exp).split('.')[0])
    mkdir_if_missing(output_dir)

    if run_idx is not None:
        output_dir = os.path.join(output_dir, 'run_{}'.format(run_idx))
        mkdir_if_missing(output_dir)

    cfg['output_dir'] = output_dir
    cfg['checkpoint'] = os.path.join(cfg['output_dir'], 'checkpoint.pth.tar')
    cfg['best_model'] = os.path.join(cfg['output_dir'], 'best_model.pth.tar')
    cfg['save_dir'] = os.path.join(cfg['output_dir'], 'predictions')
    mkdir_if_missing(cfg['save_dir'])
    cfg['log_file'] = os.path.join(cfg['output_dir'], 'logger.txt')

    return cfg


def prepare_config(cfg, root_dir, data_dir='', run_idx=None):
    # Num classes
    if cfg['train_db_name'].lower() not in ['coco', 'coco_panoptic', 'cityscapes', 'cityscapes+coco']:
        raise ValueError('Invalid train db name {}'.format(cfg['train_db_name']))

    # Paths
    output_dir = os.path.join(root_dir, cfg['train_db_name'])
    mkdir_if_missing(output_dir)

    # create a unique identifier for the run based on the current time
    if isinstance(run_idx, int) and run_idx < 0:
        from datetime import datetime
        from termcolor import colored
        now = datetime.now()
        run_idx = now.strftime("%Y%m%d_%H%M%S")
        print(colored('Using current time as run identifier: {}'.format(run_idx), 'red'))
    output_dir = os.path.join(output_dir, 'run_{}'.format(run_idx))
    mkdir_if_missing(output_dir)

    cfg['data_dir'] = data_dir
    cfg['output_dir'] = output_dir
    cfg['save_dir'] = os.path.join(cfg['output_dir'], 'predictions')
    mkdir_if_missing(cfg['save_dir'])
    cfg['log_file'] = os.path.join(cfg['output_dir'], 'logger.txt')

    return cfg, run_idx
