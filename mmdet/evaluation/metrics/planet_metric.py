# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import fiona
import itertools
import os
import json
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union
import pdb
import pycocotools.mask as mask_util
import shapely
import cv2
from affine import Affine

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP, COCOevalBuilding
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
import mmdet.utils.tanmlh_polygon_utils as polygon_utils
from mmdet.utils import tanmlh_utils
from ..functional import eval_recalls, eval_map, eval_poly_map
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh


@METRICS.register_module()
class PlanetMetric(BaseMetric):
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (100, 300, 1000).
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        sort_categories (bool): Whether sort categories in annotations. Only
            used for `Objects365V1Dataset`. Defaults to False.
        use_mp_eval (bool): Whether to use mul-processing evaluation
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False,
                 use_building_eval: bool=True,
                 mask_type: str='binary',
                 calculate_mta: bool=False,
                 calculate_iou_ciou: bool=False,
                 score_thre: float=0.5,
                 min_bbox_size: int=-1,
                 split_meta_key: Optional[str] = None,
                 out_cfg: dict = {}) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast', 'bbox_fast', 'map_fast',
                           'poly_ap_fast', 'poly_fast']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}.")

        # do class wise evaluation, default False
        self.classwise = classwise
        # whether to use multi processing evaluation, default False
        self.use_mp_eval = use_mp_eval
        self.use_building_eval = use_building_eval
        self.mask_type = mask_type

        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix
        self.calculate_mta = calculate_mta
        self.calculate_iou_ciou = calculate_iou_ciou
        self.score_thre = score_thre
        self.split_meta_key = split_meta_key

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        self.ann_file = ann_file
        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file is not None:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # 'categories' list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda i: i['id'])
                    self._coco_api.dataset['categories'] = sorted_categories
                if min_bbox_size > 0:
                    for ann in self._coco_api.dataset['annotations']:
                        x1, y1, w, h = ann['bbox']
                        if w <= min_bbox_size:
                            x1 -= (min_bbox_size - w) / 2
                            w = min_bbox_size

                        if h <= min_bbox_size:
                            y1 -= (min_bbox_size - h) / 2
                            h = min_bbox_size
                        ann['bbox'] = [x1, y1, w, h]


        else:
            self._coco_api = None

        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None
        self.out_cfg = out_cfg

    def fast_eval_recall(self,
                         results: List[dict],
                         proposal_nums: Sequence[int],
                         iou_thrs: Sequence[float],
                         logger: Optional[MMLogger] = None) -> np.ndarray:
        """Evaluate proposal recall with COCO's fast_eval_recall.

        Args:
            results (List[dict]): Results of the dataset.
            proposal_nums (Sequence[int]): Proposal numbers used for
                evaluation.
            iou_thrs (Sequence[float]): IoU thresholds used for evaluation.
            logger (MMLogger, optional): Logger used for logging the recall
                summary.
        Returns:
            np.ndarray: Averaged recall results.
        """
        gt_bboxes = []
        pred_bboxes = [result['bboxes'] for result in results]
        for i in range(len(self.img_ids)):
            ann_ids = self._coco_api.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self._coco_api.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(gt_bboxes, pred_bboxes, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels'].astype(int)
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = self.cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                if 'polygons' in result:
                    data['polygon'] = result['polygons'][i]

                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)
            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=len(annotations) +
                    1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                if ann.get('mask', None):
                    mask = ann['mask']
                    # area = mask_util.area(mask)
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask
                    # annotation['area'] = float(area)
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path)
        return converted_json_path

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # result['labels'] = np.zeros(len(pred['labels']))
            # if self.out_cfg.get('save_coco', False):
            if 'segmentations' in pred:
                result['segmentations'] = pred['segmentations']

            if 'gt_instances' in data_sample:
                result['gt_instances'] = data_sample['gt_instances']
                result['cnt_pred_gt_det'] = [len(pred['bboxes']), len(result['gt_instances']['bboxes'])]

            # encode mask to RLE
            # if 'masks' in pred:
            #     result['masks'] = encode_mask_results(
            #         pred['masks'].detach().cpu().numpy()) if isinstance(
            #             pred['masks'], torch.Tensor) else pred['masks']


            if 'proposals' in data_sample:
                result['proposals'] = data_sample['proposals']['bboxes'].cpu().numpy()

            if 'pred_sem_seg' in data_sample:
                if type(data_sample['pred_sem_seg']) == dict:
                    pred_sem_seg = data_sample['pred_sem_seg']['sem_seg'].cpu().numpy()
                else:
                    pred_sem_seg = data_sample['pred_sem_seg'].sem_seg.cpu().numpy()

                if len(data_sample['gt_instances']['masks']) == 0:
                    gt_sem_seg = np.zeros_like(pred_sem_seg)
                else:
                    gt_sem_seg = data_sample['gt_instances']['masks'].merge().to_ndarray()
                # intersect = np.logical_and(pred_sem_seg[0], gt_sem_seg).sum()
                # union = np.logical_or(pred_sem_seg[0], gt_sem_seg).sum()
                tp = (pred_sem_seg * gt_sem_seg).sum()
                tn = ((1 - pred_sem_seg) * gt_sem_seg).sum()
                fp = (pred_sem_seg * (1 - gt_sem_seg)).sum()
                fn = ((1 - pred_sem_seg) * (1 - gt_sem_seg)).sum()

                result['iou_pre_eval'] = [tp, tn, fp, fn]

            if self.split_meta_key is not None:
                assert self.split_meta_key in data_sample
                result[self.split_meta_key] = data_sample[self.split_meta_key]


            if 'tif_meta' in data_sample and self.out_cfg.get('save_results', False):
                import rasterio
                tif_meta = data_sample['tif_meta']
                out_dir = self.out_cfg['out_dir']
                mask_out_dir = os.path.join(out_dir, 'mask')
                os.makedirs(mask_out_dir, exist_ok=True)

                img_name = data_sample['img_path'].split('/')[-1].split('.')[0]

                cur_pred_mask = (pred_sem_seg[0] * 255).astype(np.uint8)
                mask_out_path = osp.join(mask_out_dir, f'{img_name}.tif')
                if self.out_cfg.get('out_size', None) is not None:
                    out_size = self.out_cfg.get('out_size', None)
                    cur_pred_mask = cv2.resize(cur_pred_mask.astype(float), out_size).astype(np.uint8)

                out_poly_scale = self.out_cfg.get('out_poly_scale', 1.)

                transform = list(tif_meta['transform'])
                transform[0] *= out_poly_scale
                transform[4] *= out_poly_scale
                transform = Affine(*transform)

                with rasterio.open(
                    mask_out_path, 'w', driver='GTiff',
                    height=cur_pred_mask.shape[0],
                    width=cur_pred_mask.shape[1],
                    count=1,
                    dtype=str(cur_pred_mask.dtype),
                    crs=tif_meta['crs'],
                    transform=transform
                ) as dst:
                    dst.write(cur_pred_mask, 1)

                if 'segmentations' in pred:
                    poly_out_dir = os.path.join(out_dir, 'poly')
                    os.makedirs(poly_out_dir, exist_ok=True)
                    poly_out_path = osp.join(poly_out_dir, f'{img_name}.geojson')

                    poly_jsons = pred['segmentations']
                    out_poly_scale = self.out_cfg.get('out_poly_scale', 1.)
                    transform = tif_meta['transform']
                    affine_matrix = np.array([
                        [transform.a, transform.b, transform.c],
                        [transform.d, transform.e, transform.f],
                        [0, 0, 1]  # Homogeneous row
                    ])

                    new_poly_jsons = []
                    for poly_json in poly_jsons:
                        new_coords = []
                        if poly_json['type'] == 'Polygon':
                            for coords in poly_json['coordinates']:
                                coords = (np.array(coords) * out_poly_scale)
                                ones_column = np.ones((coords.shape[0], 1))
                                ones_coords = np.hstack([coords, ones_column])
                                trans_coords = (ones_coords @ affine_matrix.T)[:,:2]

                                new_coords.append(trans_coords.tolist())

                            temp = dict(
                                type='Polygon',
                                coordinates=new_coords
                            )
                            new_poly_jsons.append(temp)

                    poly_jsons = new_poly_jsons

                    schema = {
                        'geometry': 'Polygon',
                        'properties': {}  # If you have properties, define them here
                    }

                    with fiona.open(poly_out_path, 'w', driver='GeoJSON',
                                    crs=tif_meta['crs'], schema=schema) as dst:

                        for polygon in poly_jsons:
                            # Write each polygon into the GeoJSON file
                            dst.write({
                                'geometry': polygon,
                                'properties': {}
                            })



            # use polygon predictions first
            if 'segmentations' in pred and self.mask_type=='polygon':
                H, W = pred['masks'].shape[1:]
                rles = []

                # dt_polygons = []
                for polygon in pred['segmentations']:
                    rle = mask_util.frPyObjects(polygon, H, W)
                    rle = mask_util.merge(rle)
                    rles.append(rle)

                    """
                    exterior = np.array(polygon[0]).reshape(-1,2).tolist()
                    interiors = [np.array(x).reshape(-1,2).tolist() for x in polygon[1:]]
                    dt_polygon = shapely.geometry.Polygon(shell=exterior, holes=interiors)
                    if dt_polygon.is_valid:
                        dt_polygons.append(dt_polygon)
                    """

                """
                gt_jsons = data_sample['gt_instances']['masks'].to_json()
                gt_polygons = [shapely.geometry.shape(x) for x in gt_jsons]
                gt_polygons = [x for x in gt_polygons if x.is_valid]
                fixed_gt_polygons = polygon_utils.fix_polygons(gt_polygons, buffer=0.0)
                fixed_dt_polygons = polygon_utils.fix_polygons(dt_polygons, buffer=0.0)
                """

                # mtas = polygon_utils.compute_polygon_contour_measures(fixed_dt_polygons, fixed_gt_polygons, sampling_spacing=2.0, min_precision=0.5, max_stretch=2)

                # binary_mask = mask_util.decode(rles[6])
                result['masks'] = rles
                result['polygons'] = pred['segmentations']
                # result['mtas'] = mtas

            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()


            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """

        # split gt and prediction list
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)
        eval_results = OrderedDict()

        if self.out_cfg.get('save_coco', False):
            coco_path = self.out_cfg['saved_coco_path']
            coco_dir = '/'.join(coco_path.split('/')[:-1])
            os.makedirs(coco_dir, exist_ok=True)
            coco_dict = self.results2coco_dict(preds)
            with open(coco_path, 'w') as f_json:
                json.dump(coco_dict, f_json)

        if self.split_meta_key is not None:
            pred_keys = [x[self.split_meta_key] for x in preds]
            unique_keys = set(pred_keys)
            unique_keys = np.sort(list(unique_keys)).tolist()
            for key in unique_keys:
                print(f'evaluating for {key}...')
                idxes = (np.array(pred_keys) == key).nonzero()[0]
                cur_preds = [preds[_] for _ in idxes]
                cur_eval_results = self.compute_metrics_single(cur_preds, key=key)
                eval_results[key] = cur_eval_results

        cur_eval_results = self.compute_metrics_single(preds)
        eval_results.update(cur_eval_results)

        return eval_results

    def compute_metrics_single(self, preds, key=''):
        eval_results = OrderedDict()
        logger: MMLogger = MMLogger.get_current_instance()
        if 'iou_pre_eval' in preds[0]:
            tps = np.array([x['iou_pre_eval'][0] for x in preds])
            tns = np.array([x['iou_pre_eval'][1] for x in preds])
            fps = np.array([x['iou_pre_eval'][2] for x in preds])
            fns = np.array([x['iou_pre_eval'][3] for x in preds])

            iou = tps.sum() / (tps.sum() + tns.sum() + fps.sum())
            precision = tps.sum() / (tps.sum() + fps.sum())
            recall = tps.sum() / (tps.sum() + tns.sum())
            oa = (tps.sum() + fns.sum()) / (tps.sum() + tns.sum() + fps.sum() + fns.sum())

            logger.info(f'{key} IoU: {iou} OA: {oa}, precision: {precision}, recall: {recall}')

            eval_results['iou'] = iou
            eval_results['precision'] = precision
            eval_results['recall'] = recall

        if 'cnt_pred_gt_det' in preds[0]:
            factor = 4.77 * 4.77 * 256 * 256 / 1e6
            cnt_pred_gt_dets = np.array([pred['cnt_pred_gt_det'] for pred in preds])
            mae = np.abs(cnt_pred_gt_dets[:,0] - cnt_pred_gt_dets[:,1]).mean() / factor
            n_ratio = cnt_pred_gt_dets[:,0].sum() / cnt_pred_gt_dets[:,1].sum()

            logger.info(f'{key} MAE per km2: {mae}    N-ratio: {n_ratio}')
            eval_results['MAE'] = mae
            eval_results['N-ratio'] = n_ratio

        if 'mtas' in preds[0]:
            mtas = []
            for pred in preds:
                mtas.extend(pred['mtas'])

            mtas = [x for x in mtas if x is not None]
            mean_mta = 0
            if len(mtas) > 0:
                mean_mta = np.array(mtas).mean()

            logger.info(f'{key} MTA: {mean_mta}')

        """
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results
        """

        for metric in self.metrics:
            logger.info(f'Evaluating {metric} {key}...')

            if metric == 'poly_ap_fast':
                det_polys = [[x['segmentations']] for x in preds]
                annotations = [x['gt_instances'] for x in preds]
                det_scores = [[x['scores']] for x in preds]
                mean_ap, ap_result = eval_poly_map(det_polys, det_scores, annotations, iou_thr=0.5)
                eval_results[f'PolyAP@50_{key}'] = mean_ap

                mean_ap, ap_result = eval_poly_map(det_polys, det_scores, annotations, iou_thr=0.3)
                eval_results[f'PolyAP@30_{key}'] = mean_ap

            if metric == 'map_fast':
                det_bboxes = [[x['bboxes']] for x in preds]
                annotations = [x['gt_instances'] for x in preds]
                mean_ap, ap_result = eval_map(det_bboxes, annotations, iou_thr=0.5)
                eval_results[f'AP@50_{key}'] = mean_ap

                mean_ap, ap_result = eval_map(det_bboxes, annotations, iou_thr=0.3)
                eval_results[f'AP@30_{key}'] = mean_ap

                continue

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == 'proposal_fast':
                temp = [dict(bboxes=x['proposals']) for x in preds]
                ar = self.fast_eval_recall(temp, self.proposal_nums, self.iou_thrs, logger=logger)

                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}_{key}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')

                log_msg = f'AR results of proposals {key}:\n' + ''.join(log_msg)
                logger.info(log_msg)
                continue

            if metric == 'bbox_fast':
                det_bboxes = [x['bboxes'] for x in preds]
                gt_bboxes = [x['gt_instances']['bboxes'].numpy() for x in preds]
                recalls = eval_recalls(gt_bboxes, det_bboxes, self.proposal_nums, self.iou_thrs, logger=logger)
                ar = recalls.mean(axis=1)

                """
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                """
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}_key'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = f'AR results of bboxes {key}:\n' + ''.join(log_msg)
                logger.info(log_msg)
                continue

            if metric == 'mta':
                mtas = polygon_utils.compute_mta(coco_eval)
                mtas = [x for x in mtas if x is not None]
                mta = np.array(mtas).mean()
                eval_results['mta'] = mta
                logger.info(f'mta: {mta}')
                continue

            if metric == 'iou_ciou':
                ious, c_ious, N_pairs = polygon_utils.compute_IoU_cIoU(
                    coco_eval.cocoDt, coco_eval.cocoGt, score_thre=self.score_thre
                )
                eval_results['iou'] = np.array(ious).mean()
                eval_results['c_iou'] = np.array(c_ious).mean()
                eval_results['N_ratio'] = N_pairs[0] / N_pairs[1]
                logger.info(f'iou: {eval_results["iou"]}, c_iou: {eval_results["c_iou"]}, N_ratio: {N_pairs[0] / N_pairs[1]}')
                continue

            """
            if self.calculate_sem_seg_iou_ciou:
                pred_sem_seg_list = [x['pred_sem_seg'] for x in preds]
                ious = polygon_utils.compute_sem_seg_IoU_cIoU(
                    pred_sem_seg_list, coco_eval.cocoGt
                )
            """

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()

        return eval_results

    def results2coco_dict(self, results):


        img_infos = []
        ann_infos = []
        ann_id = 0
        for result in results:
            img_id = result['img_id']
            img_info = dict(id=img_id)
            img_infos.append(img_info)

            bboxes = result['bboxes']
            bboxes = bbox_xyxy_to_cxcywh(torch.tensor(bboxes))

            segmentations = result['segmentations']
            scores = result['scores']

            for i in range(len(bboxes)):
                ann_info = dict(
                    id = ann_id,
                    image_id = img_id,
                    bbox = bboxes[i].tolist(),
                    segmentation = segmentations[i],
                    score = scores[i].item(),
                    category_id = 100,
                    iscrowd = 0,
                    # area = 2
                )
                ann_id += 1
                ann_infos.append(ann_info)

        coco_dict = {
            'info': {'district': 'Planet Basemap', 'description': 'Basemap data from PlanetScope captured in the 2023 quarter 2', 'contributor': 'tanmlh'},
            'categories': [{'id': 100, 'name': 'building'}],
            'images': img_infos,
            'annotations': ann_infos,
        }

        return coco_dict
