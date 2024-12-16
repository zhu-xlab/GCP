# Copyright (c) OpenMMLab. All rights reserved.
import pdb
import os
import torch
import numpy as np
from tqdm import tqdm
import math
import cv2
import scipy
import pycocotools.mask as mask_util
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .mask2former import Mask2Former
from torch import Tensor
from mmdet.structures import SampleList
from typing import Dict, List, Tuple
from mmengine.logging import print_log
import logging
import torch.nn.functional as F
from mmdet.structures import DetDataSample


@MODELS.register_module()
class PolyFormerV2(Mask2Former):

    def __init__(self, frozen_parameters=None, test_mode='normal', **kwargs):
        self.frozen_parameters = frozen_parameters
        self.test_mode = test_mode
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

    def get_crop_boxes(self, img_H, img_W, crop_size=256, stride=192):
        # prepare locations to crop
        num_rows = math.ceil((img_H - crop_size) / stride) if math.ceil(
            (img_W - crop_size) /
            stride) * stride + crop_size >= img_H else math.ceil(
                (img_H - crop_size) / stride) + 1
        num_cols = math.ceil((img_W - crop_size) / stride) if math.ceil(
            (img_W - crop_size) /
            stride) * stride + crop_size >= img_W else math.ceil(
                (img_W - crop_size) / stride) + 1

        x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
        xmin = x * stride
        ymin = y * stride

        xmin = xmin.ravel()
        ymin = ymin.ravel()
        xmin_offset = np.where(xmin + crop_size > img_W, img_W - xmin - crop_size,
                               np.zeros_like(xmin))
        ymin_offset = np.where(ymin + crop_size > img_H, img_H - ymin - crop_size,
                               np.zeros_like(ymin))
        boxes = np.stack([
            xmin + xmin_offset, ymin + ymin_offset,
            np.minimum(xmin + crop_size, img_W),
            np.minimum(ymin + crop_size, img_H)
        ], axis=1)

        return boxes

    def get_patch_weight(self, patch_size):
        choice = 1
        if choice == 0:
            step_size = (1.0 - 0.5)/(patch_size/2)
            a = np.arange(1.0, 0.5, -step_size)
            b = a[::-1]
            c = np.concatenate((b,a))
            ct = c.reshape(-1,1)
            x = ct*c
            return x
        elif choice == 1:
            min_weight = 0.5
            step_count = patch_size//4
            step_size = (1.0 - min_weight)/step_count
            a = np.ones(shape=(patch_size,patch_size), dtype=np.float32)
            a = a * min_weight
            for i in range(1, step_count + 1):
                a[i:-i, i:-i] += step_size
            a = cv2.GaussianBlur(a,(5,5),0)
            return a
        else:
            a = np.ones(shape=(patch_size,patch_size), dtype=np.float32)
            return a



    def slide_inference(self, img, batch_data_samples):

        batch_size, _, h_img, w_img = img.size()
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        h_crop_up, w_crop_up = self.test_cfg.crop_up_size
        out_h, out_w = self.test_cfg.get('out_size', (h_img, w_img))
        out_size_scale = self.test_cfg.get('out_size_scale', None)

        if out_size_scale is not None:
            out_h = round(out_h * out_size_scale)
            out_w = round(out_w * out_size_scale)


        out_crop_h, out_crop_w = self.test_cfg.out_crop_size
        seed_type = self.test_cfg.get('seed_type', 'normal')

        crop_boxes = self.get_crop_boxes(h_img, w_img, h_crop, h_stride)
        weights = torch.tensor(self.get_patch_weight(out_crop_h))

        assert batch_size == 1
        split_batch_size = 1

        num_classes = 2
        preds = torch.zeros(batch_size, out_h, out_w)
        count_mat = torch.zeros(batch_size, out_h, out_w)

        """
        preds[0, 1, 100:110, 100:110] = 1
        preds[0, 1, 200:210, 100:110] = 1
        preds[0, 1, 100:110, 200:210] = 1
        preds[0, 1, 200:210, 200:210] = 1
        return preds
        """

        for i in range(batch_size):
            selected_crop_imgs = []
            selected_crop_boxes = []
            for crop_idx, crop_box in enumerate(crop_boxes):
                start_x, start_y, end_x, end_y = crop_box
                crop_img = img[i:i+1, :, start_y:end_y, start_x:end_x]
                # if (crop_img > 0).sum() > 0:
                selected_crop_imgs.append(crop_img)
                selected_crop_boxes.append(crop_box)

            start_idx = 0
            # stop = len(selected_crop_imgs) if len(selected_crop_imgs) % split_batch_size != 0 else len(selected_crop_imgs) + split_batch_size
            # splits = list(np.arange(0, stop, split_batch_size))
            splits = list(np.arange(0, len(selected_crop_imgs), split_batch_size))
            splits.append(len(selected_crop_imgs))

            # file_str = str(img_meta[0]["filename"]).split('/')[-1]
            # box_str = '_'.join([str(x) for x in img_meta[0]["crop_boxes"]])
            cur_data_samples = DetDataSample()
            cur_data_samples.set_metainfo({
                'img_shape': (h_crop_up, w_crop_up),
                'ori_shape': (h_crop_up, w_crop_up),
                'batch_input_shape': (h_crop_up, w_crop_up)
            })

            for j in tqdm(range(len(splits) - 1), desc=f'regularizing...'):
            # for j in range(len(splits) - 1):
                cur_img = torch.cat(selected_crop_imgs[splits[j]:splits[j+1]])
                # cur_img = F.interpolate(cur_img, size=(h_crop_up, w_crop_up), mode='bilinear')

                x = self.extract_feat(cur_img)

                pred_results = self.panoptic_head.predict(x, [cur_data_samples])
                # sampled_rings = torch.cat(pred_results['sampled_rings'])
                # pred_rings = torch.cat(pred_results['pred_rings'])
                prob_map = pred_results['prob_map']

                if len(prob_map) > 0:

                    prob_map = F.interpolate(prob_map.unsqueeze(0).unsqueeze(0),
                                             size=(out_crop_h, out_crop_w), mode='bilinear')[0]

                    for crop_idx, crop_box in enumerate(selected_crop_boxes[splits[j]:splits[j+1]]):

                        if out_size_scale is not None:
                            start_x, start_y, end_x, end_y = (crop_box // (h_img  / out_h * out_size_scale)).astype(int)
                        else:
                            start_x, start_y, end_x, end_y = (crop_box // (h_img  / out_h)).astype(np.int)

                        temp = prob_map[crop_idx:crop_idx+1].cpu() * weights.view(1, out_crop_h, out_crop_w)

                        preds[i:i+1, start_y:end_y, start_x:end_x] += temp
                        count_mat[i:i+1, start_y:end_y, start_x:end_x] += 1

                # break

        binary_mask = (preds.sigmoid() > 0.5)
        img_name = batch_data_samples[0].img_path.split('/')[-1]
        out_dir = 'outputs/inria'
        out_path = os.path.join(out_dir, img_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_path, binary_mask.numpy().astype(np.uint8)[0] * 255)

        # pdb.set_trace()
        # preds = preds / (count_mat + 1e-8)
        # probs = F.softmax(preds, dim=1)

        return binary_mask


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
        if self.test_mode == 'slide_inference':
            results = self.slide_inference(batch_inputs, batch_data_samples)
        else:
            feats = self.extract_feat(batch_inputs)
            # mask_cls_results, mask_pred_results, poly_pred_results
            pred_results = self.panoptic_head.predict(
                feats, batch_data_samples)

            results_list = self.panoptic_fusion_head.predict(
                batch_data_samples,
                rescale=rescale,
                **pred_results
            )

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

            if 'vert_preds' in pred_results:
                data_sample.vert_preds = pred_results['vert_preds']

            assert 'sem_results' not in pred_results, 'segmantic ' \
                'segmentation results are not supported yet.'

        return data_samples

