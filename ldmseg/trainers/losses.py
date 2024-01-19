"""
Author: Wouter Van Gansbeke

File with loss functions
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import Optional, Dict
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist

from ldmseg.utils.utils import get_world_size
from ldmseg.utils.detectron2_utils import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)


class SegmentationLosses(nn.Module):
    def __init__(
        self,
        num_points=12544,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        ignore_label=0,
        cost_mask=1.0,
        cost_class=1.0,
        temperature=1.0,
    ):
        super().__init__()
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.ignore_label = ignore_label
        self.temperature = temperature
        self.cost_mask = cost_mask
        self.cost_class = cost_class
        self.world_size = get_world_size()

    @torch.no_grad()
    def matcher(self, outputs, targets, pred_logits=None):
        """ 
        Matcher comes from Mask2Former: https://arxiv.org/abs/2112.01527
        This function is not used by default.
        """

        bs = len(outputs)
        num_queries = outputs.shape[1]
        indices = []

        for b in range(bs):
            out_mask = outputs[b]  # [num_queries, H_pred, W_pred]
            tgt_mask = targets[b]['masks']
            if tgt_mask is None:
                indices.append(None)
                continue

            cost_class = 0
            if pred_logits is not None:
                tgt_ids = targets[b]["labels"]
                out_prob = pred_logits[b].softmax(-1)  # [num_queries, num_classes]
                cost_class = -out_prob[:, tgt_ids]
                cost_class = -out_prob.view(-1, 1)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with torch.cuda.amp.autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                cost_mask = self.matcher_sigmoid_ce_loss(out_mask, tgt_mask)
                cost_dice = self.matcher_dice_loss(out_mask, tgt_mask)

            # Final cost matrix
            C = self.cost_mask * (cost_mask + cost_dice) + self.cost_class * cost_class
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @torch.no_grad()
    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_masks(
        self,
        outputs: torch.Tensor,
        targets: Dict,
        indices=None,
    ) -> torch.tensor:
        """
        Uncertainty loss for instance segmentation as used in Mask2Former: https://arxiv.org/abs/2112.01527
        Only minor modifications (i.e., simplified matching ids for ground truth and filtering of empty masks)
        """

        # we first need to convert the targets to the format expected by the loss
        if indices is None:
            targets, indices = self.prepare_targets(targets, ignore_label=self.ignore_label)

        # filter empty masks
        masks = [t["masks"] for t in targets]
        valids = [m is not None for m in masks]
        outputs = outputs[valids]
        indices = [idx for idx, v in zip(indices, valids) if v]
        masks = [m for m in masks if m is not None]

        # skip if no masks in current batch
        num_masks = sum(len(m) for m in masks)
        if num_masks == 0:
            return outputs.sum() * 0.0
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=outputs.device)
        if dist.is_available() and dist.is_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / self.world_size, min=1).item()

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs[src_idx]
        target_masks = torch.cat([t[idx[1]] for t, idx in zip(masks, indices)])

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            # sample point_coords
            if self.oversample_ratio > 0:
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: self.calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
            else:
                point_coords = torch.rand(src_masks.shape[0], self.num_points, 2, device=src_masks.device)

            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        del src_masks
        del target_masks

        loss_mask = self.sigmoid_ce_loss(point_logits, point_labels, num_masks)
        loss_dice = self.dice_loss(point_logits, point_labels, num_masks)
        return loss_mask + loss_dice

    def dice_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks

    def matcher_dice_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    def sigmoid_ce_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor
        """
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        return loss.mean(1).sum() / num_masks

    def matcher_sigmoid_ce_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor
        """
        hw = inputs.shape[1]

        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )

        loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
            "nc,mc->nm", neg, (1 - targets)
        )

        return loss / hw

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calculates the uncertainty when using sigmoid loss.
        Defined according to PointRend: https://arxiv.org/abs/1912.08193

        Args:
            logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
                class-agnostic, where R is the total number of predicted masks in all images and C is
                the number of foreground classes. The values are logits.
        Returns:
            scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
                the most uncertain locations having the highest uncertainty score.
        """
        assert logits.shape[1] == 1
        gt_class_logits = logits.clone()
        return -(torch.abs(gt_class_logits))

    def calculate_uncertainty_seg(self, sem_seg_logits):
        """ Calculates the uncertainty when using a CE loss.
            Defined according to PointRend: https://arxiv.org/abs/1912.08193
        """
        top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
        return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)  # [B, 1, P]

    def loss_ce(
        self,
        outputs: torch.Tensor,
        targets: Dict,
        indices: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if indices is not None:
            src_masks = outputs
            masks = [t["masks"] for t in targets]
            src_idx = self._get_src_permutation_idx(indices)
            target_masks = torch.cat([t[idx[1]] for t, idx in zip(masks, indices)])
            target_masks = target_masks.bool()

            h, w = target_masks.shape[-2:]
            ce_targets = torch.full((len(targets), h, w), self.ignore_label, dtype=torch.int64, device=src_masks.device)
            for mask_idx, (x, y) in enumerate(zip(*src_idx)):
                mask_i = target_masks[mask_idx]
                ce_targets[x, mask_i] = y
            targets = ce_targets

        if masks is not None:
            targets[~masks[:, 0].bool()] = self.ignore_label

        with torch.no_grad():
            # sample point_coords
            if self.oversample_ratio > 0:
                point_coords = get_uncertain_point_coords_with_randomness(
                    outputs,
                    lambda logits: self.calculate_uncertainty_seg(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
            else:
                point_coords = torch.rand(outputs.shape[0], self.num_points, 2, device=outputs.device)

            # get gt labels
            point_labels = point_sample(
                targets.unsqueeze(1).float(),
                point_coords,
                mode='nearest',
                align_corners=False,
            ).squeeze(1).to(torch.long)

        point_logits = point_sample(
            outputs,
            point_coords,
            align_corners=False,
        )

        ce_loss = F.cross_entropy(
            point_logits / self.temperature,
            point_labels,
            reduction="mean",
            ignore_index=self.ignore_label,
        )

        return ce_loss

    def point_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        do_matching: bool = False,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        We use very effective losses to quantify the quality of the reconstruction.
        Overall loss function consists of 3 terms:
            - Cross entropy loss with uncertainty sampling
            - BCE + Dice loss with uncertainty sampling

        Based on the PointRend paper: https://arxiv.org/abs/1912.08193
        Combined CE with BCE + Dice loss
        This also works with only a vanilla CE loss but this impl. trains much faster
        """

        indices = None
        if do_matching:
            targets, _ = self.prepare_targets(targets, ignore_label=self.ignore_label)
            indices = self.matcher(outputs, targets)

        # (1) ce loss on uncertain regions
        ce_loss = self.loss_ce(outputs, targets, indices=indices, masks=masks)

        # (2) bce + dice loss for uncertain regions per object mask
        mask_loss = self.loss_masks(outputs, targets, indices=indices)
        losses = {'ce': ce_loss, 'mask': mask_loss}

        return losses

    @torch.no_grad()
    def prepare_targets(
        self,
        targets,
        ignore_label=0,
    ):

        """
        Function to convert targets to the format expected by the loss

        Args:
            targets: list[Dict]
            ignore_label: int
        """
        new_targets = []
        instance_ids = []

        for idx_t, target in enumerate(targets):
            unique_classes = torch.unique(target)
            masks = []

            # 0. target masks
            for idx in unique_classes:
                if idx == ignore_label:
                    continue
                binary_target = torch.where(target == idx, 1, 0).to(torch.float32)
                masks.append(binary_target)

            unique_classes_excl = unique_classes[unique_classes != ignore_label]
            instance_ids.append(
                (
                    unique_classes_excl.cpu(),
                    torch.arange(len(unique_classes_excl), dtype=torch.int64)
                )
            )

            new_targets.append(
                {
                    "labels": torch.full(
                        (len(masks),), 0, dtype=torch.int64, device=target.device) if len(masks) > 0 else None,
                    "masks": torch.stack(masks) if len(masks) > 0 else None,
                }
            )
        return new_targets, instance_ids
