# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.registry import MODELS
from typing import Dict


@MODELS.register_module()
class OVDTwoStageDetector(TwoStageDetector):
    def __init__(self, batch2ovd=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch2ovd = batch2ovd        # mapping from batch name to ovd name

    def run_ovd(self, inputs, data_samples, ovd_name):
        x = self.extract_feat(inputs)
        losses = dict()
        if self.with_rpn:
            with torch.no_grad():
                rpn_results_list = self.rpn_head_predict(x, data_samples)
        else:
            assert data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in data_samples
            ]
        if isinstance(ovd_name, str):
            ovd_name = [ovd_name]
        for _ovd_name in ovd_name:
            losses.update(self.roi_head.run_ovd(x, data_samples, rpn_results_list,
                                                _ovd_name, inputs))
        return losses

    def rpn_head_predict(self, x, batch_data_samples):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self.rpn_head(x)
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)
        predictions = self.rpn_head.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg, rescale=False)
        return predictions

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        if isinstance(multi_batch_inputs, dict):
            losses = super().loss(multi_batch_inputs.pop('det_batch'),
                                  multi_batch_data_samples.pop('det_batch'))
            batch_names = list(multi_batch_inputs.keys())
            for batch_name in batch_names:
                ovd_name = self.batch2ovd[batch_name]
                batch_inputs = multi_batch_inputs.pop(batch_name)
                batch_data_samples = multi_batch_data_samples.pop(batch_name)
                loss_ovd = self.run_ovd(batch_inputs,
                                        batch_data_samples,
                                        ovd_name)
                for k, v in loss_ovd.items():
                    losses.update({k+f'_{batch_name}': v})
            return losses
        else:
            return super().loss(multi_batch_inputs, multi_batch_data_samples)
