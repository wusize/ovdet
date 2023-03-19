_base_ = './faster_rcnn_r50_fpn_syncbn_180k.py'

ovd_cfg = dict(type='BaronKD',
               boxes_cache=dict(json_path='data/coco/wusize/instances_val2017_base.json',
                                start_iter=20000, ),
               use_gt=True,
               bag_weight=1.0, single_weight=0.1, use_attn_mask=False, bag_temp=30.0, single_temp=50.0,
               clip_data_preprocessor=dict(
                   type='ImgDataPreprocessor',
                   mean=[(122.7709383 - 123.675) / 58.395,
                         (116.7460125 - 116.28) / 57.12,
                         (104.09373615 - 103.53) / 57.375],
                   std=[68.5005327 / 58.395,
                        66.6321579 / 57.12,
                        70.32316305 / 57.375]),
               num_words=6, word_dim=512, words_drop_ratio=0.5,
               queue_cfg=dict(names=['clip_text_features', 'clip_image_features',
                                     'clip_word_features', 'clip_patch_features'],
                              lengths=[1024] * 4,
                              emb_dim=512, id_length=1),
               sampling_cfg=dict(shape_ratio_thr=0.25,
                                 area_ratio_thr=0.01,
                                 objectness_thr=0.85,
                                 nms_thr=0.1,
                                 topk=300,
                                 max_groups=3,
                                 max_permutations=2,
                                 alpha=3.0,
                                 cut_off_thr=0.3,
                                 base_probability=0.3,
                                 interval=-0.1,
                                 ),
               )


model = dict(
    batch2ovd=dict(det_batch='baron_kd'),    # share with the det_batch
    roi_head=dict(
        clip_cfg=dict(          # ViT-B/32
            image_encoder=dict(
                _delete_=True,
                type='CLIPViT',
                input_resolution=224,
                patch_size=32,
                width=768,
                layers=12,
                heads=12,
                output_dim=512,
                init_cfg=dict(
                    type='Pretrained',
                    prefix='visual',
                    checkpoint='checkpoints/clip_vitb32.pth')
            ),
        ),
        ovd_cfg=dict(baron_kd=ovd_cfg),
        bbox_head=dict(num_words=6)
    ),
)
