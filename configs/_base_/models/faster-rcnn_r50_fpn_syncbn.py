_base_ = 'mmdet::_base_/models/faster-rcnn_r50_fpn.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
# model settings
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        style='caffe',
        init_cfg=None),
    neck=dict(
        norm_cfg=norm_cfg,),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            norm_cfg=norm_cfg,
            num_classes=80,
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CustomCrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        )
    ),
    # model training and testing settings
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,)
    )
)
