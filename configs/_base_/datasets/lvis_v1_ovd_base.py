# dataset settings
_base_ = 'mmdet::_base_/datasets/lvis_v1_instance.py'
train_dataloader = dict(
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
        dataset=dict(
            ann_file='wusize/lvis_v1_train_base.json')
    )
)
