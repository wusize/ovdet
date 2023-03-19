# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.models import CrossEntropyLoss
from mmdet.models.losses.cross_entropy_loss import _expand_onehot_labels
from ovdet.utils import load_class_freq


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         avg_non_ignore=False, **kwargs):
    ignore_index = -100 if ignore_index is None else ignore_index

    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.size(-1), ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            # The inplace writing method will have a mismatched broadcast
            # shape error if the weight and valid_mask dimensions
            # are inconsistent such as (B,N,1) and (B,N,C).
            weight = weight * valid_mask
        else:
            weight = valid_mask

    # average loss over non-ignored elements
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = valid_mask.sum().item()

    # weighted element-wise losses
    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), reduction='none')
    if class_weight is not None:
        loss = loss * class_weight[None]
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100,
                  avg_non_ignore=False,
                  ignore_bg=False):
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    if class_weight is not None:
        mask_out = class_weight < 0.00001
        pred[:, mask_out] = -float('inf')
        if ignore_bg:
            class_weight[-1] = 0.0
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,      # still use
        reduction='none',
        ignore_index=ignore_index)

    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@MODELS.register_module()
class CustomCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, ignore_bg=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_sigmoid:
            del self.cls_criterion
            self.cls_criterion = binary_cross_entropy
        elif not self.use_mask:
            del self.cls_criterion
            self.cls_criterion = cross_entropy

        if isinstance(self.class_weight, str):
            cat_freq = load_class_freq(self.class_weight, min_count=0)
            self.class_weight = (cat_freq > 0.0).float().tolist() + [1.0]
        self.ignore_bg = ignore_bg

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            ignore_bg=self.ignore_bg,
            **kwargs)
        return loss_cls
