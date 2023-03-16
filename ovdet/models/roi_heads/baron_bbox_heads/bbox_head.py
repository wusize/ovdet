# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmengine.runner.amp import autocast
from mmdet.models.roi_heads.bbox_heads import BBoxHead


@MODELS.register_module()
class BaronBBoxHead(BBoxHead):
    def __init__(self,
                 num_words=4, word_dim=512,
                 words_drop_ratio=0.5,
                 cls_temp=50.0, cls_bias=None,
                 cls_embeddings_path='', bg_embedding='zero',
                 use_attn12_output=False,
                 *args, **kwargs):
        super(BaronBBoxHead, self).__init__(*args, **kwargs)
        self.num_words = num_words
        self.word_dim = word_dim
        self.cls_temp = cls_temp
        self.words_drop_ratio = words_drop_ratio
        self.use_attn12_output = use_attn12_output
        assert self.with_cls
        assert self.reg_class_agnostic
        assert not self.custom_cls_channels
        del self.fc_cls

        cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
        cls_predictor_cfg_.update(
            in_features=self.in_channels, out_features=num_words * word_dim)
        self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if cls_bias is None:
            self.cls_bias = 0.0
        else:
            assert self.loss_cls.use_sigmoid, \
                "cls_bias only used for sigmoid logits"
            self.cls_bias = nn.Parameter(torch.ones(1) * cls_bias)
        cls_embeddings = torch.from_numpy(
            np.load(cls_embeddings_path)).float()
        assert self.num_classes == cls_embeddings.shape[0]
        self.register_buffer('cls_embeddings', cls_embeddings)
        self.learn_bg = False
        if bg_embedding == 'zero':
            self.register_buffer('bg_embedding',
                                 torch.zeros_like(cls_embeddings[:1]))
        elif bg_embedding == 'learn':
            self.bg_embedding = nn.Linear(1, cls_embeddings.shape[1])
            self.init_cfg += [
                dict(
                    type='Xavier', distribution='uniform',
                    override=dict(name='bg_embedding')),
            ]
            self.learn_bg = True
        else:
            raise ValueError(f"{bg_embedding} not supported.")

    def pred_cls_logits(self, pseudo_words, clip_model):
        text_encoder = clip_model.text_encoder
        if pseudo_words.shape[0] == 0:
            return pseudo_words.new_zeros(0, self.num_classes + 1)
        with autocast():
            valid_mask = self._drop_word(pseudo_words.half())
            pseudo_text, end_token_ids = text_encoder.prepare_pseudo_text_tensor(
                pseudo_words.half(), valid_mask)  # add start and stop token
            if self.use_attn12_output:
                cls_features, _, _ = \
                    text_encoder.encode_pseudo_text_endk(pseudo_text, end_token_ids,
                                                         text_pe=True,
                                                         stepk=12, normalize=True)
            else:
                cls_features = \
                    text_encoder.encode_pseudo_text(pseudo_text, end_token_ids,
                                                    text_pe=True, normalize=True)
        if self.learn_bg:
            input_ones = pseudo_words.new_ones(1, 1)
            bg_embedding = self.bg_embedding(input_ones)
        else:
            bg_embedding = self.bg_embedding
        cls_embeddings = torch.cat([self.cls_embeddings, bg_embedding])
        cls_logits = self.cls_temp * cls_features @ cls_embeddings.T
        if self.training and self.loss_cls.use_sigmoid:
            cls_logits += self.cls_bias
        assert cls_logits.shape[1] == self.num_classes + 1
        return cls_logits

    def _drop_word(self, pseudo_words):
        p = self.words_drop_ratio
        num_preds, num_words, _ = pseudo_words.shape
        mask = F.dropout(pseudo_words.new_ones(num_preds, num_words),
                         p=p,
                         training=self.training)
        start_end_mask = torch.ones_like(mask[:, :1])
        # check empty
        is_empty = mask.sum(dim=-1) == 0.0
        mask[is_empty, 0] = 1.0       # TODO add random on this
        mask[mask > 0.0] = 1.0
        # add start and end token mask
        valid_mask = torch.cat([start_end_mask, mask, start_end_mask], dim=-1)

        return valid_mask

    def forward(self, x, clip_model=None):
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        pseudo_words = self.fc_cls(x).view(-1, self.num_words, self.word_dim)
        cls_score = self.pred_cls_logits(pseudo_words, clip_model)
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def vision_to_language(self, x):
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        return self.fc_cls(x).view(-1, self.num_words, self.word_dim)
