# Copyright (c) OpenMMLab. All rights reserved.
import pdb
import pycocotools.mask as mask_util
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .mask2former import Mask2Former
from torch import Tensor
from mmdet.structures import SampleList
from typing import Dict, List, Tuple
from mmengine.logging import print_log
import logging


@MODELS.register_module()
class PolyFormer(Mask2Former):

    def __init__(self, frozen_parameters=None, **kwargs):
        self.frozen_parameters = frozen_parameters
        super().__init__(**kwargs)

    def init_weights(self):

        frozen_parameters = self.frozen_parameters
        # freeze parameters by prefix
        if frozen_parameters is not None:
            print_log(f'Frozen parameters: {frozen_parameters}', logger='current', level=logging.INFO)
            for name, param in self.named_parameters():
                for frozen_prefix in frozen_parameters:
                    if frozen_prefix in name:
                        param.requires_grad = False
                if param.requires_grad:
                    print_log(f'Training parameters: {name}', logger='current', level=logging.INFO)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        feats = self.extract_feat(batch_inputs)
        mask_cls_results, mask_pred_results, poly_pred_results = self.panoptic_head.predict(
            feats, batch_data_samples)

        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results,
            mask_pred_results,
            poly_pred_results,
            batch_data_samples,
            rescale=rescale)

        # for result, poly_pred in zip(results_list, poly_pred_results):
        #     result['ins_results'].segmentations = poly_pred

        results = self.add_pred_to_datasample(batch_data_samples, results_list)
        return results

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: List[dict]) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (List[dict]): Instance segmentation, segmantic
                segmentation and panoptic segmentation results.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        for data_sample, pred_results in zip(data_samples, results_list):
            if 'pan_results' in pred_results:
                data_sample.pred_panoptic_seg = pred_results['pan_results']

            if 'ins_results' in pred_results:
                data_sample.pred_instances = pred_results['ins_results']

            assert 'sem_results' not in pred_results, 'segmantic ' \
                'segmentation results are not supported yet.'

        return data_samples

