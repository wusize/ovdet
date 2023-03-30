_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn_syncbn.py',
    '../../_base_/datasets/coco_ovd_base_ms.py',
    '../../_base_/schedules/schedule_90k.py',
    '../../_base_/iter_based_runtime.py'
]

reg_layer = [
    dict(type='Linear', in_features=1024, out_features=1024),
    dict(type='ReLU', inplace=True),
    dict(type='Linear', in_features=1024, out_features=4)
]

clip_cfg = dict(          # ViT-B/32
    type='CLIP',
    image_encoder=None,
    text_encoder=dict(
        type='CLIPTextEncoder',
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,    # also the word embedding dim
        transformer_heads=8,
        transformer_layers=12,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/clip_vitb32.pth')
    )
)

model = dict(
    type='OVDTwoStageDetector',
    rpn_head=None,
    roi_head=dict(
        type='OVDStandardRoIHead',
        clip_cfg=clip_cfg,
        bbox_head=dict(
            type='BaronShared4Conv1FCBBoxHead',
            reg_predictor_cfg=reg_layer,
            reg_class_agnostic=True,
            cls_bias=None,
            num_words=6,
            cls_temp=50.0,
            cls_embeddings_path='data/metadata/coco_clip_hand_craft_attn12.npy',
            bg_embedding='learn',
            use_attn12_output=True,
        ),
    ),
)
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='ProposalBroadcaster',
        transforms=[
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
val_dataloader = dict(
    dataset=dict(
        proposal_file='proposals/d2_proposals.pkl',
        pipeline=test_pipeline))
test_dataloader = val_dataloader
