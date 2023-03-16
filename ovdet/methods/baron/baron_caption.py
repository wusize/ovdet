import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh, bbox_overlaps
from .baron_base import BaronBase
from mmengine.structures import InstanceData, BaseDataElement
from mmengine.runner.amp import autocast
from ovdet.methods.queues import Queues
from ovdet.methods.builder import build_queue, OVD
from ovdet.models.vlms.clip.utils import tokenize_dynamic
from .utils import get_normed_boxes, SoftTargetCrossEntropy, SinePositionalEncoding


def perm_generator(seq):
    seen = set()
    length = len(seq)
    while True:
        perm = tuple(random.sample(seq, length))
        if perm not in seen:
            seen.add(perm)
            yield perm


def pseudo_permutations(seq_length, num_permutation):
    rand_perms = perm_generator(list(range(seq_length)))
    return [list(next(rand_perms)) for _ in range(num_permutation)]



@OVD.register_module()
class BaronCaption(BaronBase):
    def __init__(self,
                 norm_temp,
                 loss_weight=3.0,
                 max_caps=5,
                 **kwargs):
        super(BaronCaption, self).__init__(**kwargs)
        self.norm_temp = norm_temp
        self.loss_weight = loss_weight
        self.caption_loss = SoftTargetCrossEntropy()
        self.max_caps = max_caps

    def sample_on_topk(self, proposals, image_boxes, metainfo):
        if len(proposals) > 0:
            proposals = self.preprocess_proposals(proposals,
                                                  image_boxes,
                                                  self.sampling_cfg['shape_ratio_thr'],
                                                  self.sampling_cfg['area_ratio_thr'],
                                                  self.sampling_cfg['objectness_thr'],
                                                  self.sampling_cfg['nms_thr'])
            # sample
            num_proposals = len(proposals)
            num_samples = min(self.sampling_cfg['max_num'], num_proposals)
            sampled_idx = random.sample(list(range(num_proposals)), k=num_samples)
            proposals = proposals[sampled_idx]
        # add image boxes
        added_proposals = InstanceData(bboxes=image_boxes,
                                       labels=torch.tensor([0] * len(image_boxes), dtype=torch.int64,
                                                           device=image_boxes.device),
                                       scores=torch.ones_like(image_boxes[:, 0]))
        proposals = InstanceData.cat([proposals, added_proposals])
        normed_boxes = get_normed_boxes(proposals.bboxes,
                                        image_boxes[0])

        return BaseDataElement(proposals=proposals, normed_boxes=normed_boxes,
                               metainfo=metainfo)

    def obtain_topk_proposal(self, proposals):
        num = min(len(proposals), self.sampling_cfg['topk'])
        _, topk_inds = proposals.scores.topk(num)

        return proposals[topk_inds]

    # TODO: input topk proposals
    def sample(self, rpn_results, batch_data_sample, **kwargs):
        # when loading the caption data, the image box is taken as gt_boxes for simplicity
        image_boxes = batch_data_sample.gt_instances.bboxes
        rpn_results = self.obtain_topk_proposal(rpn_results)
        sampling_result = self.sample_on_topk(rpn_results, image_boxes, batch_data_sample.metainfo)
        return sampling_result

    def _drop_word(self, word_embeddings):
        p = self.word_dropout
        num_preds, num_words, _ = word_embeddings.shape
        mask = F.dropout(word_embeddings.new_ones(num_preds, num_words),
                         p=p,
                         training=self.training)
        # check empty
        is_empty = mask.sum(dim=-1) == 0.0
        mask[is_empty, 0] = 1.0
        mask = mask > 0.0

        return mask

    @torch.no_grad()
    def get_caption_features(self, captions, image_ids, device, clip_model):
        all_captions = []
        all_image_ids = []

        for captions_, image_id in zip(captions, image_ids):
            all_captions += captions_
            all_image_ids += [image_id] * len(captions_)

        caption_image_ids = torch.tensor(all_image_ids).to(device)
        tokens = tokenize_dynamic(all_captions, truncate=True).to(device)
        caption_features = clip_model.encode_text(tokens, normalize=True).float()
        return caption_features, caption_image_ids

    def get_losses(self, pseudo_words, clip_model, sampling_results):
        image_ids = [im.get('img_id') for im in sampling_results]
        clip_model.eval()
        caption_pseudo_words = pseudo_words
        device = caption_pseudo_words.device
        clip_caption_features, caption_img_ids = self.get_caption_features([v.get('captions')[0]
                                                                            for v in sampling_results],
                                                                           image_ids,
                                                                           device,
                                                                           clip_model)

        num_boxes_per_image = [c.get('normed_boxes').shape[0] for c in sampling_results]
        positions = bbox_xyxy_to_cxcywh(torch.cat([c.get('normed_boxes') for c in sampling_results], dim=0))
        position_embeddings = self.positional_embed(positions)

        permutations_per_image = [c.normed_boxes.shape[0] for c in sampling_results]

        num_perms_per_image = [len(b) for b in permutations_per_image]
        caption_pseudo_words = (caption_pseudo_words + position_embeddings).split(num_boxes_per_image, dim=0)
        caption_pseudo_words = [[ws[perm] for perm in perms]
                                for ws, perms in zip(caption_pseudo_words, permutations_per_image)]
        caption_pseudo_sequences = [perm for b in caption_pseudo_words for perm in b]
        words_split = [perm.shape[0] for b in caption_pseudo_words for perm in b]
        words_mask = self._drop_word(torch.cat(caption_pseudo_sequences, dim=0)).split(words_split,
                                                                                       dim=0)
        caption_pseudo_sequences = [seq.flatten(0, 1)[wm.view(-1)]
                                    for seq, wm in zip(caption_pseudo_sequences,  words_mask)]
        context_length = max([seq.shape[0] for seq in caption_pseudo_sequences])
        with autocast():
            # TODO: get local image tokens
            pseudo_text, end_token_ids = clip_model.prepare_pseudo_text(
                caption_pseudo_sequences,
                context_length=context_length + 2)  # add start and stop token
            clip_text_features = \
                clip_model.encode_pseudo_text(pseudo_text, end_token_ids,
                                              text_pe=True, normalize=True,
                                              return_word_tokens=False)

        pred_image_ids = []
        for num_perms, img_id in zip(num_perms_per_image, image_ids):
            pred_image_ids += [img_id] * num_perms
        pred_image_ids = torch.tensor(pred_image_ids).to(device)

        num_preds = clip_text_features.shape[0]
        assert sum(num_perms_per_image) == num_preds
        assert pred_image_ids.shape[0] == num_preds

        global_clip_text_features = self.queues.get_queue('clip_cap_text_features')  # add "_cap_" to avoid conflict
        contrast_clip_text_features = torch.cat([clip_text_features,
                                                 global_clip_text_features[..., :-4]], dim=0)
        contrast_clip_text_image_ids = torch.cat([pred_image_ids,
                                                  global_clip_text_features[..., -4:]], dim=0)

        global_clip_caption_features = self.queues.get_queue('clip_caption_features')
        contrast_clip_caption_features = torch.cat([clip_caption_features,
                                                    global_clip_caption_features[..., :-4]], dim=0)
        contrast_clip_caption_image_ids = torch.cat([caption_img_ids,
                                                     global_clip_caption_features[..., -4:]], dim=0)

        # matrix 0
        image_id_match_matrix = contrast_clip_text_image_ids[:, None] == contrast_clip_caption_image_ids[None]
        if self.cap_mosaic == 'concat':
            label_matrix_0 = image_id_match_matrix.float().prod(dim=-1)   # every entry should match
        else:
            label_matrix_0 = image_id_match_matrix.float().sum(dim=-1).clamp(max=1.0)
        # matrix 1
        label_matrix_1 = label_matrix_0.T

        similarity_matrix_0 = self.norm_temp * contrast_clip_text_features @ contrast_clip_caption_features.T
        similarity_matrix_1 = self.norm_temp * contrast_clip_caption_features @ contrast_clip_text_features.T

        loss_0 = self.caption_loss(similarity_matrix_0, label_matrix_0)
        loss_1 = self.caption_loss(similarity_matrix_1, label_matrix_1)

        loss = loss_0 * 0.5 + loss_1 * 0.5

        clip_caption_features_update = torch.cat([clip_caption_features,
                                                  caption_img_ids], dim=-1)

        queue_update = dict(clip_caption_features=clip_caption_features_update,
                            clip_cap_text_features=torch.cat([clip_text_features, pred_image_ids], dim=-1))
        self.queues.dequeue_and_enqueue(queue_update)

        return dict(loss_caption=loss * self.loss_weight)

    def _get_permutations_for_single_image(self, num_boxes):
        max_perms = self.sampling_cfg['max_perms']
        return pseudo_permutations(num_boxes, min(math.factorial(num_boxes), max_perms))
