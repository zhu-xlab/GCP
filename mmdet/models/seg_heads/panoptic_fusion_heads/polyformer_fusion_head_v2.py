# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
import pdb

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmdet.evaluation.functional import INSTANCE_OFFSET
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.mask import mask2bbox
from mmdet.utils import OptConfigType, OptMultiConfig
from .maskformer_fusion_head import MaskFormerFusionHead
import mmdet.utils.tanmlh_polygon_utils as polygon_utils


@MODELS.register_module()
class PolyFormerFusionHeadV2(MaskFormerFusionHead):
    """MaskFormer fusion head which postprocesses results for panoptic
    segmentation, instance segmentation and semantic segmentation."""

    def instance_postprocess(self, mask_cls: Tensor, mask_pred: Tensor, poly_pred: List) -> InstanceData:
        """Instance segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            :obj:`InstanceData`: Instance segmentation results.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        """
        max_per_image = self.test_cfg.get('max_per_image', 100)
        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # shape (num_queries * num_class, )
        labels = torch.arange(self.num_classes, device=mask_cls.device).\
            unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        scores_per_image, top_indices = scores.flatten(0, 1).topk(
            max_per_image, sorted=False)
        labels_per_image = labels[top_indices]

        query_indices = top_indices // self.num_classes
        mask_pred = mask_pred[query_indices]
        poly_pred = [poly_pred[x.item()] for x in query_indices]

        # extract things
        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]
        poly_pred = [poly_pred[x.item()] for x in is_thing.nonzero().view(-1)]
        """
        num_queries = mask_cls.shape[0]

        labels_per_image = torch.zeros(num_queries, dtype=torch.long, device=mask_cls.device)
        scores_per_image = F.softmax(mask_cls, dim=-1)[:, 0]
        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                     mask_pred_binary.flatten(1).sum(1) + 1e-6)
        det_scores = scores_per_image * mask_scores_per_image
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask2bbox(mask_pred_binary)

        results = InstanceData()
        results.bboxes = bboxes
        results.labels = labels_per_image
        results.scores = det_scores
        results.masks = mask_pred_binary
        results.segmentations = poly_pred
        return results


    def predict(self,
                batch_data_samples: SampleList,
                mask_cls_results=None,
                mask_pred_results=None,
                poly_pred_results=None,
                rescale: bool = False,
                **kwargs) -> List[dict]:
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            mask_cls_results (Tensor): Mask classification logits,
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            mask_pred_results (Tensor): Mask logits, shape
                (batch_size, num_queries, h, w).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): If True, return boxes in
                original image space. Default False.

        Returns:
            list[dict]: Instance segmentation \
                results and panoptic segmentation results for each \
                image.

            .. code-block:: none

                [
                    {
                        'pan_results': PixelData,
                        'ins_results': InstanceData,
                        # semantic segmentation results are not supported yet
                        'sem_results': PixelData
                    },
                    ...
                ]
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        panoptic_on = self.test_cfg.get('panoptic_on', True)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', False)
        assert not semantic_on, 'segmantic segmentation '\
            'results are not supported yet.'

        results = []
        for i, (mask_cls_result, mask_pred_result, poly_pred_result, meta) in enumerate(
            zip(mask_cls_results, mask_pred_results, poly_pred_results, batch_img_metas)
        ):
            # remove padding
            img_height, img_width = meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]
            if 'vert_pred_results' in kwargs:
                vert_pred_result = kwargs['vert_pred_results'][i][:, :img_height, :img_width]

            if rescale:
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]
                if 'vert_pred_results' in kwargs:
                    vert_pred_result = F.interpolate(
                        vert_pred_result[:, None],
                        size=(ori_height, ori_width),
                        mode='bilinear',
                        align_corners=False)[:, 0]

            result = dict()
            if panoptic_on:
                pan_results = self.panoptic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['pan_results'] = pan_results

            if instance_on:
                ins_results = self.instance_postprocess(
                    mask_cls_result, mask_pred_result, poly_pred_result)

                result['ins_results'] = ins_results

            if semantic_on:
                sem_results = self.semantic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['sem_results'] = sem_results

            if 'vert_pred_results' in kwargs:
                result['vert_preds'] = vert_pred_result

            if 'poly_jsons' in kwargs:
                result['poly_jsons'] = kwargs['poly_jsons']
                N = len(kwargs['poly_jsons'])
                segmentations = []
                for i, poly_json in enumerate(kwargs['poly_jsons']):
                    cur_pred_poly = polygon_utils.poly_json2coco(poly_json, scale=1.)
                    segmentations.append(cur_pred_poly)

                ins_results = InstanceData()
                ins_results.bboxes = torch.zeros(N, 4)
                ins_results.labels = torch.zeros(N, dtype=torch.long)
                ins_results.scores = torch.ones(N)
                ins_results.masks = torch.zeros(N, ori_height, ori_width, dtype=torch.bool)
                ins_results.masks[:,0,0] = 1
                ins_results.segmentations = segmentations
                result['ins_results'] = ins_results

            results.append(result)

        return results
