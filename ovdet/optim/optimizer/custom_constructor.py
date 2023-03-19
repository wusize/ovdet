# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Any, Dict, List
import torch.nn as nn
from mmengine.registry import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                               OPTIMIZERS)
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim import DefaultOptimWrapperConstructor


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class CustomOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    def __call__(self, model: nn.Module) -> OptimWrapper:
        if hasattr(model, 'module'):
            model = model.module

        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        optim_wrapper_cfg.setdefault('type', 'OptimWrapper')
        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        else:
            # set param-wise lr and weight decay recursively
            params: List = []
            self.add_params(params, model)
            # grouping parameters with the same hyper-parameters
            if self.paramwise_cfg.get('reduce_param_groups', True):
                optimizer_cfg['params'] = reduce_param_groups(params)
            else:
                optimizer_cfg['params'] = params
            # enable foreach for pytorch 1.12.0+ to speed up training
            if digit_version(TORCH_VERSION) >= digit_version('1.12.0'):
                optimizer_cfg.setdefault('foreach', True)
            else:
                optimizer_cfg.pop('foreach', None)

            optimizer = OPTIMIZERS.build(optimizer_cfg)
        optim_wrapper = OPTIM_WRAPPERS.build(
            optim_wrapper_cfg, default_args=dict(optimizer=optimizer))
        return optim_wrapper


def _expand_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform parameter groups into per-parameter structure. Later items in
    `params` can overwrite parameters set in previous items.

    Ref: https://github.com/facebookresearch/detectron2/blob/main/detectron2/solver/build.py

    Args:
        params (List[Dict[str, Any]]): The parameter groups.

    Returns:
        List[Dict[str, Any]]: List of expanded parameter groups.
    """  # noqa: E501
    ret: dict = defaultdict(dict)
    for item in params:
        assert 'params' in item
        cur_params = {x: y for x, y in item.items() if x != 'params'}
        for param in item['params']:
            ret[param].update({'params': [param], **cur_params})
    return list(ret.values())


def reduce_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reorganize the parameter groups and merge duplicated groups. The number
    of parameter groups needs to be as small as possible in order to
    efficiently use the PyTorch multi-tensor optimizer. Therefore instead of
    using a parameter_group per single parameter, we reorganize the parameter
    groups and merge duplicated groups. This approach speeds up multi-tensor
    optimizer significantly.

    Ref: https://github.com/facebookresearch/detectron2/blob/main/detectron2/solver/build.py

    Args:
        params (List[Dict[str, Any]]): The parameter groups.

    Returns:
        List[Dict[str, Any]]: The reorganized parameter groups.
    """  # noqa: E501
    params = _expand_param_groups(params)
    groups = defaultdict(
        list)  # re-group all parameter groups by their hyperparams
    for item in params:
        cur_params = tuple((x, y) for x, y in item.items() if x != 'params')
        groups[cur_params].extend(item['params'])
    ret = []
    for param_keys, param_values in groups.items():
        cur = {kv[0]: kv[1] for kv in param_keys}
        cur['params'] = param_values
        ret.append(cur)
    return ret
