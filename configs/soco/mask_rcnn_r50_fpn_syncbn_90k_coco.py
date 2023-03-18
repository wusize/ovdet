_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn_syncbn.py',
    'mmdet::_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_90k.py',
    '../_base_/iter_based_runtime.py'
]

model = dict(
    rpn_head=dict(
        type='CustomRPNHead',
        anchor_generator=dict(
            scale_major=False,      # align with detectron2
        )
    ),
    roi_head=dict(
        bbox_head=dict(
            reg_class_agnostic=False,
            num_classes=80,
        ),
        mask_head=dict(num_classes=80,
                       class_agnostic=False,)
    ),
)

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',        # amp training
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.000025),
    clip_grad=dict(max_norm=35, norm_type=2),
)
load_from = 'checkpoints/res50_fpn_soco_star_400.pth'
train_dataloader = dict(
    sampler=dict(type='InfiniteSampler'),
)
