# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
import numpy as np
from sklearn.metrics import roc_curve

from mmdet.registry import MODELS
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.losses import accuracy
from mmdet.models.utils import empty_instances, multi_apply
from mmdet.models.layers import multiclass_nms
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmdet.utils import ConfigType, InstanceList, OptMultiConfig

from .convfc_bbox_head import Shared2FCBBoxHead

@MODELS.register_module()
class STBBoxHead(Shared2FCBBoxHead):
    def __init__(self, st_cfg={}, loss_dom_score=None, **kwargs):
        self.st_cfg = st_cfg
        super().__init__(**kwargs)

        if self.st_cfg.get('do_memory_bank', False):
            memory_size = self.st_cfg.get('memory_size', 65536)
            self.pos_memory = torch.nn.Parameter(torch.zeros(memory_size))
            self.neg_memory = torch.nn.Parameter(torch.zeros(memory_size))
            self.pos_memory_idx = torch.nn.Parameter(torch.zeros(1))
            self.neg_memory_idx = torch.nn.Parameter(torch.zeros(1))
            self.pos_memory.requires_grad = False
            self.neg_memory.requires_grad = False
            self.pos_memory_idx.requires_grad = False
            self.neg_memory_idx.requires_grad = False
            self.local_iter = torch.nn.Parameter(torch.zeros(1))

        if self.st_cfg.get('domain_cfg', None) is not None:
            domain_cfg = self.st_cfg['domain_cfg']
            self.loss_dom_score = MODELS.build(loss_dom_score)
            self.domain_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Step 1: Global Average Pooling
                nn.Flatten(),  # Step 2: Flatten to shape (B, C)
                nn.Linear(domain_cfg['in_channels'], domain_cfg['out_channels']),  # Step 3: Linear layer to get shape (B, C2)
            )

        if self.st_cfg.get('dynamic_net', None) is not None:
            dynamic_net = self.st_cfg['dynamic_net']
            self.dynamic_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Step 1: Global Average Pooling
                nn.Flatten(),  # Step 2: Flatten to shape (B, C)
                nn.Linear(dynamic_net['in_channels'], dynamic_net['out_channels']),  # Step 3: Linear layer to get shape (B, C2)
            )

    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        tch_cls_score: Tensor,
                        tch_bbox_pred: Tensor,
                        rois: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None,
                        x: Optional[List[Tensor]] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        if self.st_cfg.get('do_memory_bank', True):
            self.local_iter.data += 1

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)

        losses = self.loss(
            cls_score,
            bbox_pred,
            tch_cls_score,
            tch_bbox_pred,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override,
            x=x
        )

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             tch_cls_score: Tensor,
             tch_bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None,
             x: Optional[List[Tensor]] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """
        if not self.st_cfg.get('do_memory_bank', False):
            ignore_thre = self.st_cfg.get('st_ignore_thr', 0.8)
        else:
            # ignore_thre = self.st_cfg.get('st_ignore_thr', 0.8)
            ignore_thre = 1.0
            warm_up_iter = self.st_cfg.get('warm_up_iter', 1000)

            if self.local_iter.data.item() > warm_up_iter:
                pos_mean = self.pos_memory.mean()
                neg_mean = self.neg_memory.mean()
                if pos_mean > neg_mean:
                    std_ratio = self.st_cfg.get('pos_memory_std_ratio', 0.0)
                    ignore_thre = pos_mean + self.pos_memory.std() * std_ratio

        bg_class_ind = self.num_classes

        losses = dict()

        if cls_score is not None:
            tch_fg_score = 1 - F.softmax(tch_cls_score, dim=1)[:, bg_class_ind]
            pseudo_bbox_mask = (tch_fg_score >= ignore_thre) & (labels == bg_class_ind)
            label_weights[pseudo_bbox_mask] = 0.
            # print(pseudo_bbox_mask.sum() / pseudo_bbox_mask.__len__())

            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if self.st_cfg.get('do_memory_bank', False):
            scores = F.softmax(tch_cls_score, dim=1)[:,0]
            pos_scores = scores[labels == 0]
            neg_scores = scores[labels == 1]

            if (labels == 0).sum() > 1:
                pos_mean_std = torch.stack([pos_scores.mean(), pos_scores.std()])
                self.update_memory(self.pos_memory, self.pos_memory_idx, pos_mean_std)

            if (labels == 1).sum() > 1:
                neg_mean_std = torch.stack([neg_scores.mean(), neg_scores.std()])
                self.update_memory(self.neg_memory, self.neg_memory_idx, neg_mean_std)

        if self.st_cfg.get('domain_cfg', None) is not None:
            domain_cfg = self.st_cfg['domain_cfg']
            target_score = domain_cfg.get('target_score', 0.5)
            max_alpha = domain_cfg.get('max_alpha', 5)

            scores = F.softmax(tch_cls_score, dim=1)[:,0]
            exp_scores = torch.exp(tch_cls_score)
            exp_sums = torch.exp(tch_cls_score).sum(dim=-1)

            B = x[0].shape[0]
            pred_alpha = max_alpha ** ((self.domain_head(x[-1]).sigmoid() - 0.5) * 2)

            anchor_scores = []
            for i in range(B):
                cur_labels = 1 - labels[rois[:,0].long() == i].detach()
                cur_scores = scores[rois[:,0].long() == i].detach()
                # cur_logits = tch_cls_score[rois[:,0].long() == i].detach()
                # cur_domain_scores = F.softmax(cur_logits / pred_T[i], dim=1)[:,0]

                fpr, tpr, thresholds = roc_curve(1-cur_labels.cpu().numpy(), cur_scores.cpu().numpy())
                f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr) + 1e-10)
                best_index = np.argmax(f1_scores)
                tau = thresholds[best_index]
                best_f1 = f1_scores[best_index]

                anchor_masks = (cur_scores - tau).abs() < 1e-3

                anchor_scores.append(cur_scores[anchor_masks] ** pred_alpha[i])

            anchor_scores = torch.cat(anchor_scores)
            loss_dom = self.loss_dom_score(anchor_scores, torch.zeros_like(anchor_scores) + target_score)
            losses['loss_dom'] = loss_dom

            # scores = F.softmax(tch_cls_score, dim=1)
            # self.update_memory(self.pos_memory, self.pos_memory_idx, scores[labels == 0][:,0])
            # self.update_memory(self.neg_memory, self.neg_memory_idx, scores[labels == 1][:,0])

        return losses

    def update_memory(self, memory, memory_idx, data, num_sample=512):
        if len(data) == 0:
            return

        alpha = self.st_cfg.get('alpha_memory', 0.0)

        if len(data) > num_sample:
            sample_idxes = np.random.permutation(len(data))[:num_sample]
            data = data[sample_idxes]

        written_idxes = (torch.arange(len(data)) + memory_idx.data.item()) % len(memory)
        memory[written_idxes.long()] = memory[written_idxes.long()] * alpha + (1 - alpha) * data

        memory_idx += len(data)
        memory_idx = memory_idx % len(data)

    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        bbox_preds: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        x: Optional[List[Tensor]] = None) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            rois (tuple[Tensor]): Tuple of boxes to be transformed.
                Each has shape  (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_scores (tuple[Tensor]): Tuple of box scores, each has shape
                (num_boxes, num_classes + 1).
            bbox_preds (tuple[Tensor]): Tuple of box energies / deltas, each
                has shape (num_boxes, num_classes * 4).
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`, optional): `test_cfg` of R-CNN.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                bbox_pred=bbox_preds[img_id],
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg,
                x=[xx[img_id] for xx in x]
            )
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None,
            x: Optional[List[Tensor]] = None,
    ) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels

        if self.st_cfg.get('domain_cfg', None) is not None:
            domain_cfg = self.st_cfg['domain_cfg']
            max_alpha = domain_cfg.get('max_alpha', 5)
            pred_alpha = max_alpha ** ((self.domain_head(x[-1].unsqueeze(0))[0].sigmoid() - 0.5) * 2)
            results.scores = results.scores ** pred_alpha

        return results

    def forward(self, x: Tuple[Tensor], feats: Optional[List[Tensor]]=None,
                num_roi_per_img: Optional[List[int]]=None) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        if self.dynamic_net is not None:
            B = feats[0].shape[0]
            # num_roi_per_img = len(x) // B

            weights = self.dynamic_net(feats[-1])
            dynamic_fc_cls = weights.view(B, -1, 6)[:,:,:2]
            dynamic_fc_reg = weights.view(B, -1, 6)[:,:,2:]

            cls_score = []
            bbox_pred = []
            x_cls = x_cls.split(num_roi_per_img)
            x_reg = x_reg.split(num_roi_per_img)
            for i in range(B):
                # x_cls = torch.stack(x_cls.split(num_roi_per_img))
                # x_reg = torch.stack(x_reg.split(num_roi_per_img))
                # x_cls = x_cls.view(B, num_roi_per_img, -1)
                # x_reg = x_reg.view(B, num_roi_per_img, -1)
                cur_cls_score = torch.einsum('nc,ca->na', x_cls[i], dynamic_fc_cls[i])
                cur_bbox_pred = torch.einsum('nc,ca->na', x_reg[i], dynamic_fc_reg[i])
                cls_score.append(cur_cls_score)
                bbox_pred.append(cur_bbox_pred)

            cls_score = torch.cat(cls_score)
            bbox_pred = torch.cat(bbox_pred)
            # cls_score = cls_score.reshape(B*num_roi_per_img, -1)
            # bbox_pred = bbox_pred.reshape(B*num_roi_per_img, -1)

        else:
            cls_score = self.fc_cls(x_cls) if self.with_cls else None
            bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
