# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset
from .api_wrappers import COCO
from mmengine.fileio import get_local_path
import pdb
import json
import numpy as np

import copy
import os.path as osp
from typing import List, Union

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset





@DATASETS.register_module()
class CrowdAIDataset(CocoDataset):
    """Dataset for iSAID instance segmentation.

    iSAID: A Large-scale Dataset for Instance Segmentation
    in Aerial Images.

    For more detail, please refer to "projects/iSAID/README.md"
    """

    METAINFO = dict(
        classes=('building'),
        palette=[(0, 0, 255)])

    def __init__(self, coco_res_path=None, **kwargs):
        self.coco_res_path = coco_res_path
        super().__init__(**kwargs)



    def load_data_list(self):
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)

            if self.coco_res_path is not None:
                submission_file = json.loads(open(self.coco_res_path).read())
                # coco = COCO(gt_json_path)
                coco = self.coco.loadRes(submission_file)
                self.coco = coco
                # pass

        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.getCatIds(self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        # self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        self.cat_img_map = copy.deepcopy(self.coco.catToImgs)

        img_ids = self.coco.getImgIds()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.loadImgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.getAnnIds([img_id])
            raw_ann_info = self.coco.loadAnns(ann_ids)

            if self.coco_res_path is not None:
                for x in raw_ann_info:
                    x['segmentation'] = x['polygon']

            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        # data_list = [data_list[x] for x in [1416, 1426, 1682, 1379, 751]]
        return data_list


    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            # x1, y1, w, h = ann['bbox']
            # inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            # inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))

            x1, y1 = np.array(ann['segmentation'][0]).reshape(-1,2).min(axis=0)
            x2, y2 = np.array(ann['segmentation'][0]).reshape(-1,2).max(axis=0)
            w = x2 - x1
            h = y2 - y1
            margin = 0.05
            bbox = [
                max(x1 - w * margin, 0), max(y1 - h * margin, 0),
                min(x2 + w * margin, img_info['width']),
                min(y2 + h * margin, img_info['height'])
            ]

            inter_w = max(0, min(x2, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y2, img_info['height']) - max(y1, 0))

            if len(ann['segmentation']) > 1:
                pdb.set_trace()

            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            # bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info
