# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import pdb
import torch
import numpy as np
from tqdm import tqdm

import mmengine
from mmengine import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope
from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP, COCOevalBuilding
from mmengine.structures import InstanceData
from mmengine.structures import PixelData
from mmdet.structures.mask import PolygonMasks
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from pycocotools import mask as maskUtils

from mmdet.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in coco format')
    parser.add_argument('config', help='Config of the model')
    parser.add_argument('coco', help='path to the predicted coco json file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    coco_pred = COCO(args.coco)
    valid_pred_img_ids = set([x['id'] for x in coco_pred.dataset['images']])


    out_cfg = cfg.val_evaluator[0].get('out_cfg', None)
    if out_cfg is not None:
        out_cfg['save_coco'] = False

    evaluator = Evaluator(cfg.val_evaluator)
    evaluator.dataset_meta = dataset.metainfo

    results = []
    for data in tqdm(dataset):
        data_sample = data['data_samples']
        img_id = data_sample.metainfo['img_id']
        img_shape = data_sample.metainfo['img_shape']
        continent = data_sample.metainfo.get('continent', 'All')

        if not img_id in valid_pred_img_ids:
            continue

        pred_ann_ids = coco_pred.getAnnIds(img_id)
        pred_anns = coco_pred.loadAnns(pred_ann_ids)

        pred_bboxes = torch.tensor([ann['bbox'] for ann in pred_anns])
        if len(pred_bboxes) > 0:
            pred_bboxes = bbox_cxcywh_to_xyxy(pred_bboxes)
        else:
            pred_bboxes = torch.zeros(0, 4)

        pred_scores = torch.ones(len(pred_bboxes))
        pred_labels = torch.zeros(len(pred_bboxes), dtype=torch.long)
        pred_polygons = [[np.array(x) for x in ann['segmentation']] for ann in pred_anns]
        pred_polygons = PolygonMasks(pred_polygons, img_shape[0], img_shape[1])
        pred_sem_seg = pred_polygons.merge().to_tensor(dtype=torch.long, device='cpu')
        # pred_sem_seg = pred_polygons.merge().to_ndarray()
        pixel_data = PixelData(sem_seg=pred_sem_seg)

        pred_instances = InstanceData(
            bboxes=pred_bboxes,
            scores=pred_scores,
            labels=pred_labels
        )

        gt_instances = data_sample.gt_instances
        gt_instances.bboxes = gt_instances.bboxes.tensor

        result = dict(
            gt_instances=gt_instances,
            pred_instances=pred_instances,
            pred_sem_seg=pixel_data,
            img_id=img_id,
            ori_shape=img_shape,
            continent=continent
        )
        results.append(result)

    eval_results = evaluator.offline_evaluate(results)
    print(eval_results)


if __name__ == '__main__':
    main()
