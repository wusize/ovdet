from collections import OrderedDict
import torch

arch_settings = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}


def convert_bn(blobs, state_dict, d2_name, torch_name, converted_names):
    # detectron replace bn with affine channel layer
    state_dict[torch_name + '.bias'] = blobs[d2_name + '.bias']
    state_dict[torch_name + '.weight'] = blobs[d2_name + '.weight']
    state_dict[torch_name + '.running_mean'] = blobs[d2_name + '.running_mean']
    state_dict[torch_name + '.running_var'] = blobs[d2_name + '.running_var']
    converted_names.add(d2_name + '.bias')
    converted_names.add(d2_name + '.weight')
    converted_names.add(d2_name + '.running_mean')
    converted_names.add(d2_name + '.running_var')


def convert_conv_fc(blobs, state_dict, d2_name, torch_name,
                    converted_names):
    state_dict[torch_name + '.weight'] = blobs[d2_name + '.weight']
    converted_names.add(d2_name + '.weight')
    if d2_name + '.bias' in blobs:
        state_dict[torch_name + '.bias'] = blobs[d2_name + '.bias']
        converted_names.add(d2_name + '.bias')


def convert(src, dst, depth):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # load arch_settings
    blobs = torch.load(src)['model']
    if depth not in arch_settings:
        raise ValueError('Only support ResNet-50 and ResNet-101 currently')
    block_nums = arch_settings[depth]
    # convert to pytorch style
    state_dict = OrderedDict()
    converted_names = set()
    convert_conv_fc(blobs, state_dict,
                    'backbone.bottom_up.stem.conv1',
                    'backbone.conv1',
                    converted_names)
    convert_bn(blobs, state_dict,
               'backbone.bottom_up.stem.conv1.norm',
               'backbone.bn1',
               converted_names)
    for i in range(1, len(block_nums) + 1):
        for j in range(block_nums[i - 1]):
            if j == 0:
                tar_name = f'backbone.layer{i}.{j}.downsample.0'
                src_name = f'backbone.bottom_up.res{i + 1}.{j}.shortcut'
                convert_conv_fc(blobs, state_dict, src_name,
                                tar_name, converted_names)
                tar_name = f'backbone.layer{i}.{j}.downsample.1'
                src_name = f'backbone.bottom_up.res{i + 1}.{j}.shortcut.norm'
                convert_bn(blobs, state_dict, src_name, tar_name, converted_names)

            for k in range(3):
                convert_conv_fc(blobs, state_dict,
                                f'backbone.bottom_up.res{i + 1}.{j}.conv{k+1}',
                                f'backbone.layer{i}.{j}.conv{k+1}', converted_names)
                convert_bn(blobs, state_dict,
                           f'backbone.bottom_up.res{i + 1}.{j}.conv{k+1}.norm',
                           f'backbone.layer{i}.{j}.bn{k + 1}', converted_names)

    for i in range(4):
        tar_name = f"neck.lateral_convs.{i}.conv"
        src_name = f"backbone.fpn_lateral{2 + i}"
        convert_conv_fc(blobs, state_dict, src_name, tar_name, converted_names)

        tar_name = f"neck.lateral_convs.{i}.bn"
        src_name = f"backbone.fpn_lateral{2 + i}.norm"
        convert_bn(blobs, state_dict, src_name, tar_name, converted_names)

        tar_name = f"neck.fpn_convs.{i}.conv"
        src_name = f"backbone.fpn_output{2 + i}"
        convert_conv_fc(blobs, state_dict, src_name, tar_name, converted_names)

        tar_name = f"neck.fpn_convs.{i}.bn"
        src_name = f"backbone.fpn_output{2 + i}.norm"
        convert_bn(blobs, state_dict, src_name, tar_name, converted_names)

    for i in range(4):
        src_name = f'roi_heads.box_head.conv{i+1}'
        tar_name = f'roi_head.bbox_head.shared_convs.{i}.conv'
        convert_conv_fc(blobs, state_dict, src_name, tar_name, converted_names)

        src_name = f'roi_heads.box_head.conv{i+1}.norm'
        tar_name = f'roi_head.bbox_head.shared_convs.{i}.bn'
        convert_bn(blobs, state_dict, src_name, tar_name, converted_names)

    src_name = 'roi_heads.box_head.fc1'
    tar_name = f'roi_head.bbox_head.shared_fcs.0'
    convert_conv_fc(blobs, state_dict, src_name, tar_name, converted_names)

    src_name = 'roi_heads.box_predictor.cls_score.bg_embedding'
    tar_name = 'roi_head.bbox_head.bg_embedding'  # fc
    convert_conv_fc(blobs, state_dict, src_name, tar_name, converted_names)

    src_name = 'roi_heads.box_predictor.word_pred'
    tar_name = 'roi_head.bbox_head.fc_cls'  # fc
    convert_conv_fc(blobs, state_dict, src_name, tar_name, converted_names)

    src_name = 'roi_heads.box_predictor.bbox_pred.0'
    tar_name = 'roi_head.bbox_head.fc_reg.0'
    convert_conv_fc(blobs, state_dict, src_name, tar_name, converted_names)

    src_name = 'roi_heads.box_predictor.bbox_pred.2'
    tar_name = 'roi_head.bbox_head.fc_reg.2'
    convert_conv_fc(blobs, state_dict, src_name, tar_name, converted_names)

    src_name = 'proposal_generator.rpn_head.conv'
    tar_name = 'rpn_head.rpn_conv'
    convert_conv_fc(blobs, state_dict, src_name, tar_name, converted_names)

    src_name = 'proposal_generator.rpn_head.objectness_logits.0'
    tar_name = 'rpn_head.rpn_cls.0'
    convert_conv_fc(blobs, state_dict, src_name, tar_name, converted_names)

    src_name = 'proposal_generator.rpn_head.objectness_logits.1'
    tar_name = 'rpn_head.rpn_cls.2'
    convert_conv_fc(blobs, state_dict, src_name, tar_name, converted_names)

    src_name = 'proposal_generator.rpn_head.anchor_deltas'
    tar_name = 'rpn_head.rpn_reg'
    convert_conv_fc(blobs, state_dict, src_name, tar_name, converted_names)

    tar_name = 'roi_head.clip.text_encoder'
    src_name = 'roi_heads.box_predictor.clip'

    for k, v in blobs.items():
        if src_name in k and 'visual' not in k:
            state_dict[k.replace(src_name, tar_name)] = v
            converted_names.add(k)
    for key in blobs:
        if key not in converted_names:
            print(f'Not Convert: {key}')
    torch.save(state_dict, dst)

convert('checkpoints/coco_kd_best_34.0.pth', 'checkpoints/mmdet_coco_kd_best_34.0.pth', 50)
