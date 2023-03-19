import torch
import json
import torch.nn as nn
class BoxesCache(nn.Module):
    def __init__(self, json_path, num_proposals, nms_thr=0.1, save=False):
        super(BoxesCache, self).__init__()
        with open(json_path, 'r') as f:
            images_info = json.load(f)['images']
        num_images = len(images_info)
        self.image_id2ordered_id = {info['id']: ordered_id for ordered_id, info in enumerate(images_info)}
        boxes = torch.zeros(num_images, num_proposals, 5)   # [x1, y1, x2, y2, s]
        self.register_buffer("boxes", boxes, persistent=save)
        self.num_proposals = num_proposals
        self.nms_thr = nms_thr

    @torch.no_grad()
    def update(self, image_info, proposals, score_thr):
        nms_thr = self.nms_thr
        # TODO: pull cached boxes from all devices
        image_id = image_info['image_id']
        if image_id not in self.image_id2ordered_id:
            return proposals
        ordered_id = self.image_id2ordered_id[image_id]
        image_boxes_cache = self.boxes[ordered_id]
        device = image_boxes_cache.device

        # TODO: Transform to current size
        transforms = image_info['transforms']
        cur_h, cur_w = image_info['image_size']
        cached_boxes = Boxes(torch.from_numpy(
            transforms.apply_box(image_boxes_cache[:, :4].cpu())).to(device))
        # check if in the transformed image
        cur_image_box = torch.tensor([[0., 0., cur_w, cur_h]]).to(device)
        iof = pairwise_ioa(Boxes(cur_image_box), cached_boxes)
        iof[iof < 0.3] = 0.0      # not in the current image
        cached_box_scores = iof.view(-1) * image_boxes_cache[:, 4]
        cached_boxes.clip((cur_h, cur_w))   # clip out of bound areas

        proposal_boxes = proposals.proposal_boxes.tensor
        proposal_scores = proposals.objectness_logits.sigmoid()

        merged_boxes = torch.cat([cached_boxes.tensor, proposal_boxes])
        merged_scores = torch.cat([cached_box_scores, proposal_scores])

        score_kept = merged_scores > score_thr
        if score_kept.sum() == 0:
            score_kept = [0]

        merged_boxes = merged_boxes[score_kept]
        merged_scores = merged_scores[score_kept]

        nmsed_kept = nms(merged_boxes, merged_scores, nms_thr)

        kept_boxes = merged_boxes[nmsed_kept]
        kept_scores = merged_scores[nmsed_kept]

        out = Instances(image_size=proposals.image_size,
                        proposal_boxes=Boxes(kept_boxes),
                        objectness_logits=inverse_sigmoid(kept_scores))
        if not self.training:
            return out
        # TODO: transform to the original size
        proposal_boxes = torch.from_numpy(
            transforms.inverse().apply_box(
                proposal_boxes.cpu())
        ).to(device)

        merged_boxes = torch.cat([image_boxes_cache[:, :4], proposal_boxes])
        merged_scores = torch.cat([image_boxes_cache[:, 4], proposal_scores])
        score_kept = merged_scores > score_thr
        if score_kept.sum() == 0:
            score_kept = [0]

        merged_boxes = merged_boxes[score_kept]
        merged_scores = merged_scores[score_kept]

        nmsed_kept = nms(merged_boxes, merged_scores, nms_thr)

        kept_boxes = merged_boxes[nmsed_kept]
        kept_scores = merged_scores[nmsed_kept]

        num_update = min(self.num_proposals, len(kept_boxes))
        device = kept_scores.device
        update_cache_to_sync = torch.zeros(self.num_proposals, 6)     # [x,y,x,y,s,order_id]
        update_cache_to_sync[:, -1] = float(ordered_id)    # ordered_id
        update_cache = torch.cat([kept_boxes, kept_scores[:, None]], dim=1)[:num_update]
        update_cache_to_sync[:num_update, :5] = update_cache

        # sync for updates from other devices
        update_cache = comm.all_gather(update_cache_to_sync)
        for update_cache_ in update_cache:
            ordered_id_ = int(update_cache_[0, -1].item())
            self.boxes[ordered_id_] = update_cache_[:, :5].to(device)  # update

        return out
