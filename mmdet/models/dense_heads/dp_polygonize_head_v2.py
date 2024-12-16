# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple, Union
import pdb
import rasterio
import shapely
import numpy as np
from rasterio.features import shapes
import pycocotools.mask as mask_util
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmengine.model import ModuleList, caffe2_xavier_init
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean, InstanceList
from mmdet.registry import MODELS, TASK_UTILS
from ..layers import Mask2FormerTransformerDecoder, SinePositionalEncoding, PolyFormerTransformerDecoder
import mmdet.utils.tanmlh_polygon_utils as polygon_utils

from ..utils import get_point_coords_around_ring, get_point_coords_around_ring_v2

@MODELS.register_module()
class DPPolygonizeHeadV2(nn.Module):

    def __init__(self, poly_cfg, decoder=None, feat_channels=256, loss_dice_wn=None,
                 loss_poly_reg=None):
        super().__init__()

        self.poly_cfg = poly_cfg

        self.positional_encoding = SinePositionalEncoding(num_feats=128, normalize=True)
        self.decoder = None
        if decoder is not None:
            self.decoder = Mask2FormerTransformerDecoder(**decoder)
            self.num_decoder_layers = decoder.num_layers
            self.poly_reg_head = nn.Sequential(
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, 2)
            )
            self.poly_embed = nn.Linear(2, feat_channels)
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='PointL1Cost', weight=1.),
                ]
            )
            self.assigner = TASK_UTILS.build(assigner)
            self.feat_channels = feat_channels

            self.loss_dice_wn = MODELS.build(loss_dice_wn)
            self.loss_poly_reg = MODELS.build(loss_poly_reg)

    def loss(
        self, pred_jsons, gt_jsons, W, device='cpu', points_coords=None,
        point_targets=None, mask_targets=None, **kwargs
    ):

        assert len(pred_jsons) == len(gt_jsons)
        N = self.poly_cfg.get('num_inter_points', 96)
        K = len(pred_jsons)

        sampled_rings, _, _ = polygon_utils.sample_rings_from_json(
            pred_jsons, interval=self.poly_cfg.get('step_size'), only_exterior=True,
            num_min_bins=self.poly_cfg.get('num_min_bins', 8),
            num_bins=self.poly_cfg.get('num_bins', None)
        )
        sampled_segments, is_complete = polygon_utils.sample_segments_from_rings(sampled_rings, self.poly_cfg.get('num_inter_points'))

        prim_reg_preds = torch.zeros(K, N, 2, device=device)
        prim_reg_targets = torch.zeros(K, N, 2, device=device)

        sampled_segments = sampled_segments.to(device)

        poly_pred_results = self.forward(sampled_segments, W, **kwargs)

        prim_reg_pred = poly_pred_results['prim_reg_pred']

        losses = dict()

        match_idxes = []
        for i in range(K):
            prim_target = self._get_poly_targets_single(
                prim_reg_pred[i].detach().cpu(), gt_jsons[i],
                sampled_segments=sampled_segments[i].cpu()
            )
            prim_reg_targets[i] = prim_target['prim_reg_targets']
            if is_complete[i]:
                seg_mask = (sampled_segments[i] >= 0).all(dim=-1)
                pred_poly = shapely.geometry.Polygon(sampled_segments[i][seg_mask].tolist())
                gt_poly = shapely.geometry.Polygon(gt_jsons[i]['coordinates'][0])
                iou = polygon_utils.polygon_iou(pred_poly, gt_poly)
                if iou > self.poly_cfg.get('align_iou_thre', 0.5):
                    match_idxes.append(i)

        match_idxes = torch.tensor(match_idxes)

        sizes = (prim_reg_pred >= 0).all(dim=-1).sum(dim=1)

        # decoded_rings = polygon_utils.batch_decode_ring_dp(prim_reg_pred, sizes, max_step_size=64, lam=4, device=prim_reg_pred.device)
        if self.poly_cfg.get('apply_poly_iou_loss', False):
            # prim_reg_pred = torch.cat([prim_reg_pred, prim_reg_pred[:, :1]], dim=1)
            dp, dp_points = polygon_utils.batch_decode_ring_dp(
                prim_reg_pred, sizes, max_step_size=sizes.max(),
                lam=self.poly_cfg.get('lam', 4),
                device=device, return_both=True,
                result_device=device
            )
        else:
            dp = polygon_utils.batch_decode_ring_dp(
                prim_reg_pred, sizes, max_step_size=sizes.max(),
                lam=self.poly_cfg.get('lam', 4),
                device=device, only_return_dp=True
            )

        # opt_dis = torch.gather(dp[:,0], 1, sizes.unsqueeze(1)-1)

        opt_dis_comp = torch.gather(dp[is_complete], 2, sizes[is_complete].unsqueeze(1).unsqueeze(1).repeat(1,N,1)).min(dim=1)[0]
        opt_dis_incomp = torch.gather(dp[~is_complete, 0], 1, sizes[~is_complete].unsqueeze(1)-1)

        losses['loss_dp'] = (opt_dis_comp.sum() + opt_dis_incomp.sum()) / K * self.poly_cfg.get('loss_weight_dp', 0.01)

        dp_points = [x[:-1] for x in dp_points]
        if self.poly_cfg.get('apply_poly_iou_loss', False):

            if len(match_idxes) > 0:
                if points_coords is None:
                    # sample points and point_targets
                    sampled_coords = []
                    # point_targets = []
                    for i in match_idxes:
                        pred_ring = dp_points[i]
                        # sampled_coord = get_point_coords_around_ring(pred_ring, W)
                        temp = prim_reg_pred[i][(prim_reg_pred[i] >= 0).all(dim=-1)]
                        sampled_coord = get_point_coords_around_ring_v2(
                            temp.detach(), W, num_samples=1024,
                            max_offsets=self.poly_cfg.get('max_sample_offsets', 5)
                        )
                        # point_target = mask_targets[i, sampled_coord[:,1], sampled_coord[:,0]]

                        sampled_coords.append(sampled_coord)
                        # point_targets.append(point_target)
                    sampled_coords = torch.stack(sampled_coords, dim=0) if len(sampled_coords) > 0 else torch.zeros(0, 1024, 2, device=prim_reg_pred.device)
                    point_targets = point_sample(mask_targets[match_idxes].float().unsqueeze(1), sampled_coords / W, align_corners=False).squeeze(1)

                else:
                    pdb.set_trace()
                    sampled_coords = points_coords[match_idxes] * W
                    point_targets = point_targets[match_idxes]

                loss_poly_iou = prim_reg_pred[:0].sum()
                wn_list = []
                gt_wn_list = []
                # for i, (pred_ring, sampled_coords) in enumerate(zip(dp_points, points_coords)):
                for i, idx in enumerate(match_idxes):
                    pred_ring = dp_points[idx]
                    # gt_ring = torch.tensor(gt_jsons[idx]['coordinates'][0], device=pred_ring.device)[:-1]
                    wn = polygon_utils.cal_winding_number(pred_ring, sampled_coords[i], c=1)
                    # gt_wn = polygon_utils.cal_winding_number(gt_ring, sampled_coords[i], c=1) > 0.5
                    # gt_wn = polygon_utils.cal_winding_number(gt_ring, sampled_coords[i], c=1)

                    wn_list.append(wn)
                    # gt_wn_list.append(gt_wn)

                """
                vis_data = {
                    'polygons_and_points': dict(
                        polygons=[shapely.geometry.Polygon(dp_points[x].tolist()) for x in match_idxes],
                        points=[x.detach().cpu().numpy() for x in sampled_coords],
                        point_labels=[(x > 0.5).long().cpu().numpy() for x in wn_list]
                    )
                }
                polygon_utils.vis_data_wandb(vis_data)
                """

                wns = torch.stack(wn_list, dim=0)
                # gt_wns = torch.cat(point_targets, dim=0).unsqueeze(1)
                # gt_wns = torch.stack(gt_wn_list, dim=0)
                # ((torch.cat(gt_wn_list) > 0.5) == gt_wns[:,0]).sum()
                loss_poly_iou = self.loss_dice_wn(wns, point_targets)
                # loss_poly_iou = self.loss_dice_wn(wns, gt_wns)
                # pdb.set_trace()
                # loss_poly_iou = (wns - gt_wns).abs().mean()
                losses['loss_poly_iou'] = loss_poly_iou
            else:
                losses['loss_poly_iou'] = prim_reg_pred[:0].sum()

        # Polygon regression
        A = prim_reg_pred.reshape(-1, 2)
        B = prim_reg_targets.view(-1, 2)

        if self.poly_cfg.get('reg_targets_type', 'vertice') == 'contour':
            mask = (poly_pred >= 0).all(dim=-1).view(-1)
            loss_poly_reg = self.loss_poly_reg(A[mask], B[mask])
        elif self.poly_cfg.get('reg_targets_type', 'vertice') == 'vertice':
            mask = (prim_reg_targets >= 0).all(dim=-1).view(-1)
            loss_poly_reg = self.loss_poly_reg(A[mask], B[mask])
        else:
            raise ValueError()

        losses['loss_poly_reg'] = loss_poly_reg
        if self.poly_cfg.get('use_right_angle_loss', False):
            mask = (prim_ind_targets == 0)
            loss_right_angle = prim_reg_pred[:0].sum()
            if K > 0:
                for i in range(K):
                    if mask[i].sum() > 0:
                        cur_pred = prim_reg_pred[i, mask[i]]
                        cur_target = prim_reg_targets[i, mask[i]]
                        angle_pred, valid_mask = polygon_utils.calculate_polygon_angles(cur_pred)
                        angle_target, valid_mask2 = polygon_utils.calculate_polygon_angles(cur_target)
                        valid_mask = valid_mask & valid_mask2
                        loss_right_angle += self.loss_poly_right_ang(angle_pred[valid_mask], angle_target[valid_mask])
                loss_right_angle = loss_right_angle / K

            losses['loss_poly_right_ang'] = loss_right_angle

        return losses

    def forward(self, poly_pred, W, mask_feat=None, query_feat=None, batch_idxes=None):

        results = dict()

        K, N, _ = poly_pred.shape
        C = self.feat_channels

        norm_poly_pred = (poly_pred / W - 0.5) * 2
        poly_valid_mask = (poly_pred >= 0).all(dim=-1)
        poly_feat = self.poly_embed(norm_poly_pred).view(K, N, C)

        if mask_feat is not None:
            point_feat_list = []
            for i, cur_mask_feat in enumerate(mask_feat):
                cur_norm_poly_pred = norm_poly_pred[batch_idxes == i].unsqueeze(0)

                point_feat = F.grid_sample(
                    cur_mask_feat[None], cur_norm_poly_pred, align_corners=True
                )
                point_feat = point_feat.permute(0,2,3,1).squeeze(0)
                point_feat_list.append(point_feat)

            point_feat = torch.cat(point_feat_list, dim=0)
            poly_feat += point_feat

            if self.poly_cfg.get('use_decoded_feat_in_poly_feat', False):
                poly_feat += query_feat.detach().view(K, 1, -1)

        poly_pos_embed = self.positional_encoding(poly_feat.new_zeros(K, N, 1))
        poly_pos_embed = poly_pos_embed.view(K, C, N).permute(0,2,1)
        # poly_pos_embed += ((torch.arange(N, device=poly_pred.device) / N - 0.5) * 2).view(1,-1,1)

        query_feat = poly_feat
        query_embed = poly_pos_embed

        prim_pred_reg_list = []
        for i in range(self.num_decoder_layers):
            layer = self.decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=poly_feat,
                value=poly_feat,
                query_pos=query_embed,
                key_pos=poly_pos_embed,
                cross_attn_mask=None,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)

            if i == self.num_decoder_layers - 1:

                prim_pred_reg = self.poly_reg_head(query_feat).view(K, N, -1)
                prim_pred_reg_list.append(prim_pred_reg)

        prim_pred_reg = prim_pred_reg_list[-1]

        prim_pred_reg = poly_pred + prim_pred_reg * self.poly_cfg.get('max_offsets', 10)
        prim_pred_reg = torch.clamp(prim_pred_reg, 0, W)
        prim_pred_reg[(poly_pred < 0).all(dim=-1)] = -1

        results['prim_reg_pred'] = prim_pred_reg

        return results

    def _get_poly_targets_single(self, poly_pred, poly_gt_json, sampled_segments):

        targets = {}

        N = self.poly_cfg.get('num_inter_points', 96)
        max_align_dis = self.poly_cfg.get('max_align_dis', 1e8)

        prim_reg_targets = torch.zeros(N, 2) - 1
        prim_ind_targets = torch.zeros(N, dtype=torch.long)
        prim_ref_targets = torch.zeros(N, 2) - 1

        K = (sampled_segments >= 0).all(dim=-1).sum()

        poly_gt_torch = torch.tensor(poly_gt_json['coordinates'][0]).float() # use the exterior

        if K == 0 or (poly_gt_torch == 0).all():
            targets['prim_ind_targets'] = prim_ind_targets
            targets['prim_reg_targets'] = prim_reg_targets
            return targets

        gt_instances = InstanceData(
            labels=torch.zeros(len(poly_gt_torch[:-1]), dtype=torch.long),
            points=poly_gt_torch[:-1]
        ) # (num_classes, N)

        pred_instances = InstanceData(points=sampled_segments[:K])

        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=None)

        gt_inds = assign_result.gt_inds
        seg_inds = gt_inds.nonzero().view(-1)
        gt_inds = gt_inds[seg_inds]

        dis = ((poly_gt_torch[gt_inds - 1] - sampled_segments[seg_inds]) ** 2).sum(dim=1) ** 0.5
        max_align_dis = self.poly_cfg.get('max_align_dis', 1e8)
        valid_mask = dis < max_align_dis

        prim_reg_targets[seg_inds[valid_mask]] = poly_gt_torch[gt_inds[valid_mask] - 1]
        prim_ind_targets[seg_inds[valid_mask]] = 1

        targets['prim_ind_targets'] = prim_ind_targets
        targets['prim_reg_targets'] = prim_reg_targets

        return targets


