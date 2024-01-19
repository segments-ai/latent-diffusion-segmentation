"""
Author: Wouter Van Gansbeke

File with helper functions
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

"""

import os
import sys
import errno
from typing import Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from detectron2.utils.visualizer import (
    Visualizer, _PanopticPrediction,
    ColorMode, _OFF_WHITE, _create_text_labels
)


class OutputDict(OrderedDict):
    # TODO: change to NamedTuple
    # keep setitem and setattr in sync
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# helper functions for distributed training
def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def has_batchnorms(model: nn.Module):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for _, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def gpu_gather(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        tensor = tensor.clone()[None]
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    return torch.cat(output_tensors, dim=0)


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    start_warmup_value: int = 0,
    warmup_iters: Optional[int] = None,
) -> np.ndarray:
    """ Cosine scheduler with warmup.
    """

    warmup_schedule = np.array([])
    if warmup_iters is None:
        warmup_iters = 0
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def warmup_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    start_warmup_value: int = 0,
    warmup_iters: Optional[int] = None,
) -> np.ndarray:
    """ Linear warmup scheduler.
    """

    warmup_schedule = np.array([])
    if warmup_iters is None:
        warmup_iters = 0
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def step_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    decay_epochs: list = [20, 40],
    decay_rate: float = 0.1,
    start_warmup_value: int = 0,
    warmup_iters: Optional[int] = None,
) -> np.ndarray:
    """ Step scheduler with warmup.
    """
    assert isinstance(decay_epochs, list), "decay_epochs must be a list"

    warmup_schedule = np.array([])
    if warmup_iters is None:
        warmup_iters = 0
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
    for decay_epoch in decay_epochs:
        schedule[int(decay_epoch * niter_per_ep - warmup_iters):] *= decay_rate
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            if not os.path.exists(os.path.dirname(fpath)):
                os.makedirs(os.path.dirname(fpath))
                self.file = open(fpath, 'w')
            else:
                self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def color_map(N: int = 256, normalized: bool = False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def collate_fn(batch: dict):
    # TODO: make general
    images = torch.stack([d['image'] for d in batch])
    semseg = torch.stack([d['semseg'] for d in batch])
    image_semseg = torch.stack([d['image_semseg'] for d in batch])
    tokens = mask = inpainting_mask = text = meta = None
    if 'tokens' in batch[0]:
        tokens = torch.stack([d['tokens'] for d in batch])
    if 'mask' in batch[0]:
        mask = torch.stack([d['mask'] for d in batch])
    if 'inpainting_mask' in batch[0]:
        inpainting_mask = torch.stack([d['inpainting_mask'] for d in batch])
    if 'text' in batch[0]:
        text = [d['text'] for d in batch]
    if 'meta' in batch[0]:
        meta = [d['meta'] for d in batch]
    return {
        'image': images,
        'semseg': semseg,
        'meta': meta,
        'text': text,
        'tokens': tokens,
        'mask': mask,
        'inpainting_mask': inpainting_mask,
        'image_semseg': image_semseg
    }


class MyVisualizer(Visualizer):
    def draw_panoptic_seg(
        self,
        panoptic_seg,
        segments_info,
        area_threshold=None,
        alpha=0.7,
        random_colors=False,
        suppress_thing_labels=False,
    ):
        """
        Only minor changes to the original function from detectron2.utils.visualizer
        """

        pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(self._create_grayscale_image(pred.non_empty_mask()))

        # draw mask for all semantic segments first i.e. "stuff"
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata["stuff_colors"][category_idx]]
            except AttributeError:
                mask_color = None

            text = self.metadata["stuff_classes"][category_idx]
            self.draw_binary_mask(
                mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )

        # draw mask for all instances second
        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return self.output
        masks, sinfo = list(zip(*all_instances))
        category_ids = [x["category_id"] for x in sinfo]

        try:
            scores = [x["score"] for x in sinfo]
        except KeyError:
            scores = None
        class_names = self.metadata["thing_classes"] if not suppress_thing_labels else ["object"] * 2
        labels = _create_text_labels(
            category_ids, scores, class_names, [x.get("iscrowd", 0) for x in sinfo]
        )

        try:
            colors = [
                self._jitter([x / 255 for x in self.metadata["thing_colors"][c]]) for c in category_ids
            ]
        except AttributeError:
            colors = None

        if random_colors:
            colors = None
        self.overlay_instances(masks=masks, labels=labels, assigned_colors=colors, alpha=alpha)

        return self.output


def get_imagenet_stats():
    stats = {
        'mean': torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        'std': torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        'mean_clip': torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
        'std_clip': torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
    }
    return stats
