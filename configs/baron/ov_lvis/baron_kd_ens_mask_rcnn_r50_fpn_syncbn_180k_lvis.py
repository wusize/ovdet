_base_ = './baron_kd_mask_rcnn_r50_fpn_syncbn_180k_lvis.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='EnsembleBaronShared4Conv1FCBBoxHead',
            ensemble_factor=2.0 / 3.0,
            class_info='data/metadata/lvis_v1_train_cat_norare_info.json',
            transfer_factor=None
        ),
    ),
)
