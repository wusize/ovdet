# Copyright (c) OpenMMLab. All rights reserved.
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

from mmengine.registry import HOOKS
from mmengine.hooks.hook import Hook
from mmengine.logging import print_log


@HOOKS.register_module()
class FP16CompressionHook(Hook):
    priority = 'VERY_HIGH'

    def before_train(self, runner) -> None:
        if runner.distributed:
            print_log('==========register fp16 compress hook============')
            runner.model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
