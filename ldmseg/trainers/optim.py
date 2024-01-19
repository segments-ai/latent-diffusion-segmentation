"""
Author: Wouter Van Gansbeke

# File with optimizers for training
# Parts of this file are based on detectron2 (Apache License): https://github.com/facebookresearch/detectron2
"""

import torch
from termcolor import colored
from typing import Any, Dict
from typing import List, Tuple, Optional, Callable, Set
import copy
from collections import defaultdict


def get_optim(
    model: torch.nn.Module,
    base_lr: float = 0.0001,
    weight_decay: float = 0.0,
    weight_decay_norm: float = 0.0,
    betas: Tuple = (0.9, 0.999),
    lr_factor_func: Optional[Callable] = None,
    zero_redundancy: bool = False,
    verbose: bool = True,
    save_optim: Optional[bool] = None,
) -> Tuple[torch.optim.Optimizer, bool]:

    # TODO: make `overrides`` an optional argument

    ret = get_optimizer_params(
        model, weight_decay=None, weight_decay_norm=weight_decay_norm, base_lr=base_lr,
        lr_factor_func=lr_factor_func,
        overrides={
            'module.backbone.image_encoder.net.pos_embed': {'weight_decay': 0.0},
            'module.backbone_image.image_encoder.net.pos_embed': {'weight_decay': 0.0},
        },
        verbose=verbose,
    )
    if zero_redundancy:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        optim = ZeroRedundancyOptimizer(
            ret, optimizer_class=torch.optim.AdamW,
            lr=base_lr, weight_decay=weight_decay, betas=betas,
        )
        save_optim = False if save_optim is None else save_optim
    else:
        optim = torch.optim.AdamW(ret, lr=base_lr, weight_decay=weight_decay, betas=betas)
        save_optim = True if save_optim is None else save_optim
    return optim, save_optim


def get_optim_unet(
    model: torch.nn.Module,
    base_lr: float = 0.0001,
    weight_decay: float = 0.0,
    weight_decay_norm: float = 0.0,
    betas: Tuple = (0.9, 0.999),
    lr_factor_func: Optional[Callable] = None,
    zero_redundancy: bool = False,
    verbose: bool = True,
    save_optim: Optional[bool] = None,
) -> Tuple[torch.optim.Optimizer, bool]:

    ret = get_optimizer_params(
        model, weight_decay=None, weight_decay_norm=weight_decay_norm, base_lr=base_lr,
        lr_factor_func=lr_factor_func,
        overrides={'module.object_queries.weight': {'weight_decay': 0.0}},
        verbose=verbose,
    )
    if zero_redundancy:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        optim = ZeroRedundancyOptimizer(
            ret, optimizer_class=torch.optim.AdamW,
            lr=base_lr, weight_decay=weight_decay, betas=betas,
        )
        save_optim = False if save_optim is None else save_optim
    else:
        optim = torch.optim.AdamW(ret, lr=base_lr, weight_decay=weight_decay, betas=betas)
        save_optim = True if save_optim is None else save_optim
    return optim, save_optim


def get_optim_general(
    params: Dict[str, Any],
    name: str,
    p: Dict[str, Any],
    zero_redundancy: bool = False,
    save_optim: Optional[bool] = None,
) -> Tuple[torch.optim.Optimizer, bool]:

    """ Returns the optimizer to be used for training
    """
    if name != 'adamw8bit' and not zero_redundancy:
        print(colored('Most likely not enough memory for training', 'red'))
        print(colored(f'Please use zero redundancy optimizer with {name}', 'red'))

    if 'weight_decay_norm' in p.keys():
        del p['weight_decay_norm']

    if zero_redundancy:
        # reinitialize optimizer with ZeroRedundancyOptimizer
        from torch.distributed.optim import ZeroRedundancyOptimizer
        if name == 'adamw':
            optim_cls = torch.optim.AdamW
        elif name == 'adamw8bit':
            import bitsandbytes as bnb
            optim_cls = bnb.optim.AdamW8bit
        elif name == 'sgd':
            optim_cls = torch.optim.SGD
            if 'momentum' not in p.keys():
                p['momentum'] = 0.9
            if 'betas' in p.keys():
                del p['betas']
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented with zero redundancy')

        optimizer = ZeroRedundancyOptimizer(
            params,
            optimizer_class=optim_cls,
            **p,
        )
        save_optim = False if save_optim is None else save_optim
        print(colored('Using ZeroRedundancyOptimizer w/o storing optim to save memory', 'red'))
        return optimizer, save_optim

    if name == 'adamw':
        optimizer = torch.optim.AdamW(params, **p)
    elif name == 'adamw8bit':
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(params, **p)
    elif name == 'adam':
        optimizer = torch.optim.Adam(params, **p)
    elif name == 'sgd':
        if 'momentum' not in p.keys():
            p['momentum'] = 0.9
        if 'betas' in p.keys():
            del p['betas']
        optimizer = torch.optim.SGD(params, **p)
    else:
        raise NotImplementedError(f'Unknown optimizer {name}')

    save_optim = True if save_optim is None else save_optim
    return optimizer, save_optim


def get_optimizer_params(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    lr_factor_func: Optional[Callable] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:

    # Based on the implementation from detectron2
    # TODO: clean this up

    if overrides is None:
        overrides = {}
    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        if base_lr is None:
            raise ValueError("bias_lr_factor requires base_lr")
        bias_overrides["lr"] = base_lr * bias_lr_factor
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias
    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides
    if lr_factor_func is not None:
        if base_lr is None:
            raise ValueError("lr_factor_func requires base_lr")
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
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm
            if lr_factor_func is not None:
                hyperparams["lr"] *= lr_factor_func(f"{module_name}.{module_param_name}")

            hyperparams.update(overrides.get(f"{module_name}.{module_param_name}", {}))
            params.append({"params": [value], **hyperparams})
            if verbose:
                print(f'Adding {module_name}.{module_param_name} to optimizer with {hyperparams}')
    return reduce_param_groups(params)


def _expand_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # copied from detectron2
    ret = defaultdict(dict)
    for item in params:
        assert "params" in item
        cur_params = {x: y for x, y in item.items() if x != "params"}
        for param in item["params"]:
            ret[param].update({"params": [param], **cur_params})
    return list(ret.values())


def reduce_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # copied from detectron2
    params = _expand_param_groups(params)
    groups = defaultdict(list)  # re-group all parameter groups by their hyperparams
    for item in params:
        cur_params = tuple((x, y) for x, y in item.items() if x != "params")
        groups[cur_params].extend(item["params"])
    ret = []
    for param_keys, param_values in groups.items():
        cur = {kv[0]: kv[1] for kv in param_keys}
        cur["params"] = param_values
        ret.append(cur)
    return ret
