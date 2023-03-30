_base_ = [
    'mmdet::_base_/models/rpn_r50_fpn.py',
    '../../_base_/datasets/coco_ovd_base_ms.py',
    '../../_base_/schedules/schedule_90k.py',
    '../../_base_/iter_based_runtime.py'
]

model = dict(
    neck=dict(norm_cfg=dict(type='BN'),),
    rpn_head=dict(
        type='DetachRPNHead',
        anchor_generator=dict(
            scale_major=False,      # align with detectron2
        )
    ),
)
data_root = 'data/coco/'
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'wusize/instances_val2017_base.json',
        metric='proposal_fast',
        prefix='Base',
        format_only=False),
    dict(
        type='CocoMetric',
        ann_file=data_root + 'wusize/instances_val2017_novel.json',
        metric='proposal_fast',
        prefix='Novel',
        format_only=False)
]
test_evaluator = val_evaluator
