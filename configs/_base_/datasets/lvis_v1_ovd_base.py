# dataset settings
_base_ = 'mmdet::_base_/datasets/lvis_v1_instance.py'
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            ann_file='annotations/lvis_v1_train_norare.json')
    )
)
