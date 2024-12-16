# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import warnings
from datetime import datetime
import os
import cv2
import mmcv
import numpy as np
import pdb
import shapely
import geopandas as gpd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None
import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData
from mmengine.visualization import Visualizer

from ..evaluation import INSTANCE_OFFSET
from ..registry import VISUALIZERS
from ..structures import DetDataSample
from ..structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from .palette import _get_adaptive_scales, get_palette, jitter_color

from mmengine.visualization.utils import (check_type, check_type_and_length,
                                          color_str2rgb, color_val_matplotlib,
                                          convert_overlay_heatmap,
                                          img_from_canvas, tensor2ndarray,
                                          value2list, wait_continue)


@VISUALIZERS.register_module()
class TanmlhVisualizer(Visualizer):
    """MMDetection Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str, tuple(int), optional): Color of texts.
            The tuple of color should be in BGR order.
            Defaults to (200, 200, 200).
        mask_color (str, tuple(int), optional): Color of masks.
            The tuple of color should be in BGR order.
            Defaults to None.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
            Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import InstanceData
        >>> from mmdet.structures import DetDataSample
        >>> from mmdet.visualization import DetLocalVisualizer

        >>> det_local_visualizer = DetLocalVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_instances = InstanceData()
        >>> gt_instances.bboxes = torch.Tensor([[1, 2, 2, 5]])
        >>> gt_instances.labels = torch.randint(0, 2, (1,))
        >>> gt_det_data_sample = DetDataSample()
        >>> gt_det_data_sample.gt_instances = gt_instances
        >>> det_local_visualizer.add_datasample('image', image,
        ...                         gt_det_data_sample)
        >>> det_local_visualizer.add_datasample(
        ...                       'image', image, gt_det_data_sample,
        ...                        out_file='out_file.jpg')
        >>> det_local_visualizer.add_datasample(
        ...                        'image', image, gt_det_data_sample,
        ...                         show=True)
        >>> pred_instances = InstanceData()
        >>> pred_instances.bboxes = torch.Tensor([[2, 4, 4, 8]])
        >>> pred_instances.labels = torch.randint(0, 2, (1,))
        >>> pred_det_data_sample = DetDataSample()
        >>> pred_det_data_sample.pred_instances = pred_instances
        >>> det_local_visualizer.add_datasample('image', image,
        ...                         gt_det_data_sample,
        ...                         pred_det_data_sample)
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8,
                 draw_bbox=True) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.mask_color = mask_color
        self.line_width = line_width
        self.alpha = alpha
        self.draw_bbox = draw_bbox
        # Set default value. When calling
        # `DetLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    def _draw_instances(self, image: np.ndarray, instances: ['InstanceData'],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances and instances.bboxes.sum() > 0 and self.draw_bbox:
            bboxes = instances.bboxes
            labels = instances.labels

            max_label = int(max(labels) if len(labels) > 0 else 0)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]

            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)

            positions = bboxes[:, :2] + self.line_width
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = ''
                # if 'label_names' in instances:
                #     label_text = instances.label_names[i]
                # else:
                #     label_text = classes[
                #         label] if classes is not None else f'class {label}'
                if 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    # label_text += f': {score}'
                    label_text += f'{score}'

                self.draw_texts(
                    label_text,
                    pos,
                    colors=text_colors[i],
                    font_sizes=int(13 * scales[i]),
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        if 'masks' in instances:
            labels = instances.labels
            masks = instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            masks = masks.astype(bool)

            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None \
                else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            colors = [jitter_color(mask_palette[label]) for label in labels]
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)

            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)

            if len(labels) > 0 and \
                    ('bboxes' not in instances or
                     instances.bboxes.sum() == 0):
                # instances.bboxes.sum()==0 represent dummy bboxes.
                # A typical example of SOLO does not exist bbox branch.
                areas = []
                positions = []
                for mask in masks:
                    _, _, stats, centroids = cv2.connectedComponentsWithStats(
                        mask.astype(np.uint8), connectivity=8)
                    if stats.shape[0] > 1:
                        largest_id = np.argmax(stats[1:, -1]) + 1
                        positions.append(centroids[largest_id])
                        areas.append(stats[largest_id, -1])
                areas = np.stack(areas, axis=0)
                scales = _get_adaptive_scales(areas)

                for i, (pos, label) in enumerate(zip(positions, labels)):
                    label_text = ''
                    # if 'label_names' in instances:
                    #     label_text = instances.label_names[i]
                    # else:
                    #     label_text = classes[
                    #         label] if classes is not None else f'class {label}'

                    if 'scores' in instances:
                        score = round(float(instances.scores[i]) * 100, 1)
                        # label_text += f': {score}'
                        label_text += f'{score}'

                    self.draw_texts(
                        label_text,
                        pos,
                        colors=text_colors[i],
                        font_sizes=int(13 * scales[i]),
                        horizontal_alignments='center',
                        bboxes=[{
                            'facecolor': 'black',
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])

        return self.get_image()

    def _draw_panoptic_seg(self, image: np.ndarray,
                           panoptic_seg: ['PixelData'],
                           classes: Optional[List[str]],
                           palette: Optional[List]) -> np.ndarray:
        """Draw panoptic seg of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            panoptic_seg (:obj:`PixelData`): Data structure for
                pixel-level annotations or predictions.
            classes (List[str], optional): Category information.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        # TODO: Is there a way to bypass？
        num_classes = len(classes)

        panoptic_seg_data = panoptic_seg.sem_seg[0]

        ids = np.unique(panoptic_seg_data)[::-1]

        if 'label_names' in panoptic_seg:
            # open set panoptic segmentation
            classes = panoptic_seg.metainfo['label_names']
            ignore_index = panoptic_seg.metainfo.get('ignore_index',
                                                     len(classes))
            ids = ids[ids != ignore_index]
        else:
            # for VOID label
            ids = ids[ids != num_classes]

        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = (panoptic_seg_data[None] == ids[:, None, None])

        max_label = int(max(labels) if len(labels) > 0 else 0)

        mask_color = palette if self.mask_color is None \
            else self.mask_color
        mask_palette = get_palette(mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]

        self.set_image(image)

        # draw segm
        polygons = []
        for i, mask in enumerate(segms):
            contours, _ = bitmap_to_polygon(mask)
            polygons.extend(contours)
        self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
        self.draw_binary_masks(segms, colors=colors, alphas=self.alpha)

        # draw label
        areas = []
        positions = []
        for mask in segms:
            _, _, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8)
            max_id = np.argmax(stats[1:, -1]) + 1
            positions.append(centroids[max_id])
            areas.append(stats[max_id, -1])
        areas = np.stack(areas, axis=0)
        scales = _get_adaptive_scales(areas)

        text_palette = get_palette(self.text_color, max_label + 1)
        text_colors = [text_palette[label] for label in labels]

        for i, (pos, label) in enumerate(zip(positions, labels)):
            label_text = classes[label]

            self.draw_texts(
                label_text,
                pos,
                colors=text_colors[i],
                font_sizes=int(13 * scales[i]),
                bboxes=[{
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                }],
                horizontal_alignments='center')
        return self.get_image()

    def _draw_sem_seg(self, image: np.ndarray, sem_seg: PixelData,
                      classes: Optional[List],
                      palette: Optional[List]) -> np.ndarray:
        """Draw semantic seg of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            sem_seg (:obj:`PixelData`): Data structure for pixel-level
                annotations or predictions.
            classes (list, optional): Input classes for result rendering, as
                the prediction of segmentation model is a segment map with
                label indices, `classes` is a list which includes items
                responding to the label indices. If classes is not defined,
                visualizer will take `cityscapes` classes by default.
                Defaults to None.
            palette (list, optional): Input palette for result rendering, which
                is a list of color palette responding to the classes.
                Defaults to None.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        sem_seg_data = sem_seg.sem_seg
        if isinstance(sem_seg_data, torch.Tensor):
            sem_seg_data = sem_seg_data.numpy()

        # 0 ~ num_class, the value 0 means background
        ids = np.unique(sem_seg_data)
        ignore_index = sem_seg.metainfo.get('ignore_index', 255)
        ids = ids[ids != ignore_index]

        if 'label_names' in sem_seg:
            # open set semseg
            label_names = sem_seg.metainfo['label_names']
        else:
            label_names = classes

        labels = np.array(ids, dtype=np.int64)
        colors = [palette[label] for label in labels]

        self.set_image(image)

        # draw semantic masks
        for i, (label, color) in enumerate(zip(labels, colors)):
            masks = sem_seg_data == label
            self.draw_binary_masks(masks, colors=[color], alphas=self.alpha)
            label_text = label_names[label]
            _, _, stats, centroids = cv2.connectedComponentsWithStats(
                masks[0].astype(np.uint8), connectivity=8)
            if stats.shape[0] > 1:
                largest_id = np.argmax(stats[1:, -1]) + 1
                centroids = centroids[largest_id]

                areas = stats[largest_id, -1]
                scales = _get_adaptive_scales(areas)

                self.draw_texts(
                    label_text,
                    centroids,
                    colors=(255, 255, 255),
                    font_sizes=int(13 * scales),
                    horizontal_alignments='center',
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        return self.get_image()

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            # draw_gt: bool = False,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            gt_img_data = image
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances(image,
                                                   data_sample.gt_instances,
                                                   classes, palette)

            """
            if 'gt_sem_seg' in data_sample:
                gt_img_data = self._draw_sem_seg(gt_img_data,
                                                 data_sample.gt_sem_seg,
                                                 classes, palette)

            if 'gt_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                gt_img_data = self._draw_panoptic_seg(
                    gt_img_data, data_sample.gt_panoptic_seg, classes, palette)
            """

        if draw_pred and data_sample is not None:
            pred_img_data = image
            self.add_image('ori_image', image, step)
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr]

                pred_img_data = self._draw_instances(image, pred_instances,
                                                     classes, palette)

                self.add_image('pred_instance', pred_img_data, step)

                if 'segmentations' in pred_instances:
                    vis_poly = self._vis_poly(image, pred_instances.segmentations)
                    if vis_poly is not None:
                        pred_img_data = np.concatenate((pred_img_data, vis_poly), axis=1)
                        self.add_image('pred_poly', vis_poly, step)

                if draw_gt and 'gt_instances' in data_sample:
                    # try:
                    vis_gt_poly = self._vis_poly(image, data_sample.gt_instances.masks.to_json())
                    if vis_gt_poly is not None:
                        pred_img_data = np.concatenate((pred_img_data, vis_gt_poly), axis=1)
                        self.add_image('gt_poly', vis_gt_poly, step)
                    # except:
                    #     pdb.set_trace()


            if 'poly_reg_targets' in data_sample:
                poly_reg_targets = data_sample.poly_reg_targets
                vis_poly = self._vis_poly(image, poly_reg_targets)
                pred_img_data = np.concatenate((pred_img_data, vis_poly), axis=1)

            if 'prim_reg_targets' in data_sample:
                prim_reg_targets = data_sample.prim_reg_targets
                rings = []
                for ring in prim_reg_targets:
                    mask = (ring >= 0).all(dim=-1)
                    cur_ring = ring[mask][:-1].tolist()
                    if len(cur_ring) >= 3:
                        rings.append([cur_ring])

                vis_poly = self._vis_poly(image, rings)
                pred_img_data = np.concatenate((pred_img_data, vis_poly), axis=1)

            if 'vert_preds' in data_sample:
                vert_preds = data_sample.vert_preds
                vert_featmap = self.draw_featmap(vert_preds.max(dim=0)[0][None, :])
                pred_img_data = np.concatenate((pred_img_data, vert_featmap), axis=1)


            if 'pred_sem_seg' in data_sample:
                vis_sem_seg = self._draw_sem_seg(image, data_sample.pred_sem_seg, classes, palette)
                pred_img_data = np.concatenate((pred_img_data, vis_sem_seg), axis=1)
            """

            if 'pred_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_panoptic_seg(
                    pred_img_data, data_sample.pred_panoptic_seg.numpy(),
                    classes, palette)
            """

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)

    def _vis_poly(self, img, polygons, alpha_face=0.2, alpha_edge=0.5, alpha_point=1., alpha=0.0):
        H, W, C = img.shape
        if C == 3:
            img = img[:,:,[2,1,0]]

        if type(polygons) == list:
            if len(polygons) == 0:
                polygons_shape = []

            elif type(polygons[0]) == list: # coco format
                polygons_shape = []
                for polygon in polygons:
                    coords = []
                    for x in polygon:
                        coords.append(np.array(x).reshape(-1,2).tolist())

                    polygons_shape.append(shapely.geometry.Polygon(shell=coords[0], holes=coords[1:]))

            elif type(polygons[0]) == dict: # json format
                polygons_shape = []
                for polygon in polygons:
                    polygons_shape.append(shapely.geometry.shape(polygon))

        elif type(polygons) == torch.Tensor:
            polygons_shape = []
            for polygon in polygons:
                coords = polygon.tolist()
                polygons_shape.append(shapely.geometry.Polygon(shell=coords))


        polygons = polygons_shape

        gdf = gpd.GeoDataFrame({'geometry': polygons})

        num_colors = len(gdf)
        colors = plt.cm.Spectral(np.linspace(0, 1, num_colors))
        np.random.shuffle(colors)
        face_colors = colors.copy()
        edge_colors = colors.copy()
        point_colors = colors.copy()

        face_colors[:,-1] = alpha_face
        edge_colors[:,-1] = alpha_edge
        point_colors[:,-1] = alpha_point
        # edge_colors = np.array([[0.,0.,1.,.5]] * len(colors))

        ax = gdf.plot(color=face_colors, edgecolor=edge_colors, linewidth=1)
        # ax = gdf.plot(color=face_colors, edgecolor=edge_colors, linewidth=W // 4)
        ax.imshow(img)

        for i, polygon in enumerate(polygons):
            if polygon.geom_type == 'Polygon':
                rings = [polygon.exterior, *polygon.interiors]
            elif polygon.geom_type == 'MultiPolygon':
                rings_list = []
                for poly in polygon.geoms:
                    rings = [poly.exterior, *poly.interiors]
                    rings_list.extend(rings)

                rings = rings_list
            else:
                pdb.set_trace()

            coords = [ring.xy for ring in rings]
            for xi, yi in coords:
                # ax.plot(xi[:-1], yi[:-1], marker="o", color='blue', markersize=W // 2)
                # ax.plot(xi[:-1], yi[:-1], marker="o", color=point_colors[i], markersize=W)
                ax.plot(xi[:-1], yi[:-1], marker="o", color=point_colors[i], markersize=2)
                # ax.plot(xi[:-1], yi[:-1], marker="o", color=point_colors[i], markersize=W//2)
                # ax.plot(xi[:3], yi[:3], marker="o", color='red', markersize=W)
                # ax.plot(xi[4:7], yi[4:7], marker="o", color='green', markersize=W)
                # ax.plot(xi, yi, marker="o", color='red', markersize=W // 2)

        # Customizing plot - removing axes and setting size
        ax.set_axis_off()  # Remove axes
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_position([0, 0, 1, 1])
        ax.invert_yaxis()

        fig = ax.figure
        fig.set_size_inches(10.24, 10.24)  # Change the size of the figure

        # plt.tight_layout(pad=0)
        temp_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plt.savefig(f'.{temp_name}.png', dpi=100)
        vis_img = cv2.imread(f'.{temp_name}.png')
        os.remove(f'.{temp_name}.png')

        if vis_img is not None:
            mask = (vis_img == 255).all(axis=-1)
            # vis_img = cv2.resize(vis_img, img.shape[:2], interpolation=cv2.INTER_NEAREST)
            vis_img = cv2.resize(vis_img, img.shape[:2])
            # mask = cv2.resize(mask.astype(np.uint8), img.shape[:2], interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask.astype(np.uint8), img.shape[:2])

            vis_img = np.where(np.expand_dims(mask, 2), img, vis_img * (1 - alpha) + img * alpha)
            # vis_img = vis_img * (1 - alpha) + img * alpha
            vis_img = vis_img.clip(0, 255).astype(np.uint8)

        plt.close()
        return vis_img

    def draw_bboxes(
        self,
        bboxes: Union[np.ndarray, torch.Tensor],
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
        face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
        alpha: Union[int, float] = 0.8,
    ) -> 'Visualizer':
        """Draw single or multiple bboxes.

        Args:
            bboxes (Union[np.ndarray, torch.Tensor]): The bboxes to draw with
                the format of(x1,y1,x2,y2).
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'g'.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors. Defaults to None.
            alpha (Union[int, float]): The transparency of bboxes.
                Defaults to 0.8.
        """
        check_type('bboxes', bboxes, (np.ndarray, torch.Tensor))

        if len(bboxes.shape) == 1:
            bboxes = bboxes[None]

        if bboxes.shape[-1] == 4:
            bboxes = tensor2ndarray(bboxes)
            assert (bboxes[:, 0] <= bboxes[:, 2]).all() and (bboxes[:, 1] <=
                                                             bboxes[:, 3]).all()
            if not self._is_posion_valid(bboxes.reshape((-1, 2, 2))):
                warnings.warn(
                    'Warning: The bbox is out of bounds,'
                    ' the drawn bbox may not be in the image', UserWarning)
            poly = np.stack(
                (bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 1],
                 bboxes[:, 2], bboxes[:, 3], bboxes[:, 0], bboxes[:, 3]),
                axis=-1).reshape(-1, 4, 2)
            poly = [p for p in poly]
        elif bboxes.shape[-1] == 5:
            # rotated boxes
            from mmdet.structures.bbox import mmrotate_transforms
            poly = mmrotate_transforms.obb2poly(bboxes, 'oc').reshape(-1,4,2)
            poly = tensor2ndarray(poly)
            poly = [p for p in poly]

        else:
            raise ValueError('The format of bbox should either be in (x1,y1,x2,y2) or (cx,cy,w,h,a)')

        return self.draw_polygons(
            poly,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=face_colors)

def random_color(seed):
    """Random a color according to the input seed."""
    if sns is None:
        raise RuntimeError('motmetrics is not installed,\
                 please install it by: pip install seaborn')
    np.random.seed(seed)
    colors = sns.color_palette()
    color = colors[np.random.choice(range(len(colors)))]
    color = tuple([int(255 * c) for c in color])
    return color

