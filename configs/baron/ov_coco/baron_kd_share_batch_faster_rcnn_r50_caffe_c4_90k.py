_base_ = './faster_rcnn_r50_caffe_c4_90k.py'
class_weight = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 0, 1, 1, 0, 1, 0, 0, 1,
                0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                1, 0, 1, 1, 1, 1, 0, 0, 0, 1] + [1]
ovd_cfg = dict(type='BaronKD',
               boxes_cache=None,
               use_gt=True,
               bag_weight=1.0, single_weight=0.1, use_attn_mask=False,
               bag_temp=30.0, single_temp=50.0,
               clip_data_preprocessor=dict(
                   type='ImgDataPreprocessor',
                   bgr_to_rgb=True,
                   mean=[122.7709383 - 123.675,
                         116.7460125 - 116.28,
                         104.09373615 - 103.53],
                   std=[68.5005327, 66.6321579, 70.32316305]),
               num_words=4, word_dim=512, words_drop_ratio=0.5,
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
        bbox_head=dict(num_words=4,
                       cls_bias=None,
                       cls_temp=50.0,
                       bg_embedding='learn',
                       loss_cls=dict(
                           type='CustomCrossEntropyLoss',
                           use_sigmoid=False,
                           class_weight=class_weight),
                       )

    ),
)
