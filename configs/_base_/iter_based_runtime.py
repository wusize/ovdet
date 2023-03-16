_base_ = 'mmdet::_base_/default_runtime.py'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=100000000)
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
find_unused_parameters = True
