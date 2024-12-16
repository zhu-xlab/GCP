# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Type, TypeVar, Union
import pdb

import cv2
import numpy as np
import torch
from torch import BoolTensor, Tensor

from mmdet.structures.mask.structures import BitmapMasks, PolygonMasks
from .horizontal_boxes import HorizontalBoxes
from .base_boxes import BaseBoxes
from .bbox_overlaps import bbox_overlaps
from .box_type import register_box
from . import mmrotate_transforms

T = TypeVar('T')
DeviceType = Union[str, torch.device]
MaskType = Union[BitmapMasks, PolygonMasks]


@register_box(name='rbox')
class RotatedBoxes(HorizontalBoxes):
    """The horizontal box class used in MMDetection by default.

    The ``box_dim`` of ``HorizontalBoxes`` is 4, which means the length of
    the last dimension of the data should be 4. Two modes of box data are
    supported in ``HorizontalBoxes``:

    - 'xyxy': Each row of data indicates (x1, y1, x2, y2), which are the
      coordinates of the left-top and right-bottom points.
    - 'cxcywh': Each row of data indicates (x, y, w, h), where (x, y) are the
      coordinates of the box centers and (w, h) are the width and height.

    ``HorizontalBoxes`` only restores 'xyxy' mode of data. If the the data is
    in 'cxcywh' mode, users need to input ``in_mode='cxcywh'`` and The code
    will convert the 'cxcywh' data to 'xyxy' automatically.

    Args:
        data (Tensor or np.ndarray or Sequence): The box data with shape of
            (..., 4).
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
        mode (str, Optional): the mode of boxes. If it is 'cxcywh', the
            `data` will be converted to 'xyxy' mode. Defaults to None.
    """

    box_dim: int = 5

    def __init__(self,
                 data: Union[Tensor, np.ndarray, Sequence],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[DeviceType] = None,
                 clone: bool = True,
                 in_mode='cxcywha',
                 version='oc') -> None:

        self.version = version
        self.in_mode=in_mode

        if isinstance(data, (np.ndarray, Tensor, Sequence)):
            data = torch.as_tensor(data)
        else:
            raise TypeError('boxes should be Tensor, ndarray, or Sequence, ',
                            f'but got {type(data)}')

        if device is not None or dtype is not None:
            data = data.to(dtype=dtype, device=device)
        # Clone the data to avoid potential bugs
        if clone:
            data = data.clone()
        # handle the empty input like []
        if data.numel() == 0:
            data = data.reshape((-1, self.box_dim))

        assert data.dim() >= 2 and data.size(-1) == self.box_dim, \
            ('The boxes dimension must >= 2 and the length of the last '
             f'dimension must be {self.box_dim}, but got boxes with '
             f'shape {data.shape}.')

        if in_mode not in ('cxcywha'):
            raise ValueError(f'Get invalid mode {in_mode}.')

        self.tensor = data

    def translate_(self, distances: Tuple[float, float]) -> None:
        """Translate boxes in-place.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        boxes = self.tensor
        assert len(distances) == 2
        self.tensor[:,:2] = boxes[:,:2] + boxes.new_tensor(distances).reshape(-1,2)

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Geometric transformat boxes in-place.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
        """
        boxes = self.tensor

        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = boxes.new_tensor(homography_matrix)

        corners = mmrotate_transforms.obb2poly(boxes, self.version)
        corners = corners.reshape(len(corners), -1, 2)

        # corners = self.hbox2corner(boxes)
        corners = torch.cat([corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        corners = corners.reshape(len(corners), 8)
        self.tensor = mmrotate_transforms.poly2obb(corners, self.version)


    def flip_(self,
              img_shape: Tuple[int, int],
              direction: str = 'horizontal') -> None:
        """Flip boxes horizontally or vertically in-place.
        Copied from MMRotate

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        """

        assert direction in ['horizontal', 'vertical', 'diagonal']
        bboxes = self.tensor.clone()
        self.tensor = mmrotate_transforms.bbox_flip(bboxes, img_shape, direction=direction)

    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        """Rotate all boxes in-place.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degrees. Positive
                values mean clockwise rotation.
        """
        boxes = self.tensor
        rotation_matrix = boxes.new_tensor(
            cv2.getRotationMatrix2D(center, -angle, 1))

        pdb.set_trace()
        corners = self.hbox2corner(boxes)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(rotation_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        self.tensor = self.corner2hbox(corners)

    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        """Rescale boxes w.r.t. rescale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
        """
        boxes = self.tensor
        assert len(scale_factor) == 2
        scale_factor = boxes.new_tensor(list(scale_factor) + [np.sqrt(scale_factor[0] * scale_factor[1])] * 2)
        self.tensor[:,:4] = boxes[:,:4] * scale_factor.reshape(1,4)

    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        """Resize the box width and height w.r.t scale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.
        """
        pdb.set_trace()
        boxes = self.tensor
        assert len(scale_factor) == 2
        ctrs = (boxes[..., 2:] + boxes[..., :2]) / 2
        wh = boxes[..., 2:] - boxes[..., :2]
        scale_factor = boxes.new_tensor(scale_factor)
        wh = wh * scale_factor
        xy1 = ctrs - 0.5 * wh
        xy2 = ctrs + 0.5 * wh
        self.tensor = torch.cat([xy1, xy2], dim=-1)

    def is_inside(self,
                  img_shape: Tuple[int, int],
                  all_inside: bool = False,
                  allowed_border: int = 0) -> BoolTensor:
        """Find boxes inside the image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            all_inside (bool): Whether the boxes are all inside the image or
                part inside the image. Defaults to False.
            allowed_border (int): Boxes that extend beyond the image shape
                boundary by more than ``allowed_border`` are considered
                "outside" Defaults to 0.
        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside
            the image. Assuming the original boxes have shape (m, n, 4),
            the output has shape (m, n).
        """
        img_h, img_w = img_shape
        boxes = self.tensor
        if all_inside:
            return (boxes[:, 0] >= -allowed_border) & \
                (boxes[:, 1] >= -allowed_border) & \
                (boxes[:, 2] < img_w + allowed_border) & \
                (boxes[:, 3] < img_h + allowed_border)
        else:
            return (boxes[..., 0] < img_w + allowed_border) & \
                (boxes[..., 1] < img_h + allowed_border) & \
                (boxes[..., 2] > -allowed_border) & \
                (boxes[..., 3] > -allowed_border)

    def find_inside_points(self,
                           points: Tensor,
                           is_aligned: bool = False) -> BoolTensor:
        """Find inside box points. Boxes dimension must be 2.

        Args:
            points (Tensor): Points coordinates. Has shape of (m, 2).
            is_aligned (bool): Whether ``points`` has been aligned with boxes
                or not. If True, the length of boxes and ``points`` should be
                the same. Defaults to False.

        Returns:
            BoolTensor: A BoolTensor indicating whether a point is inside
            boxes. Assuming the boxes has shape of (n, 4), if ``is_aligned``
            is False. The index has shape of (m, n). If ``is_aligned`` is
            True, m should be equal to n and the index has shape of (m, ).
        """
        boxes = self.tensor
        assert boxes.dim() == 2, 'boxes dimension must be 2.'

        if not is_aligned:
            boxes = boxes[None, :, :]
            points = points[:, None, :]
        else:
            assert boxes.size(0) == points.size(0)

        x_min, y_min, x_max, y_max = boxes.unbind(dim=-1)
        return (points[..., 0] >= x_min) & (points[..., 0] <= x_max) & \
            (points[..., 1] >= y_min) & (points[..., 1] <= y_max)

    def create_masks(self, img_shape: Tuple[int, int]) -> BitmapMasks:
        """
        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            :obj:`BitmapMasks`: Converted masks
        """
        img_h, img_w = img_shape
        boxes = self.tensor

        xmin, ymin = boxes[:, 0:1], boxes[:, 1:2]
        xmax, ymax = boxes[:, 2:3], boxes[:, 3:4]
        gt_masks = np.zeros((len(boxes), img_h, img_w), dtype=np.uint8)
        for i in range(len(boxes)):
            gt_masks[i,
                     int(ymin[i]):int(ymax[i]),
                     int(xmin[i]):int(xmax[i])] = 1
        return BitmapMasks(gt_masks, img_h, img_w)

    @staticmethod
    def overlaps(boxes1: BaseBoxes,
                 boxes2: BaseBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False,
                 eps: float = 1e-6) -> Tensor:
        """Calculate overlap between two set of boxes with their types
        converted to ``HorizontalBoxes``.

        Args:
            boxes1 (:obj:`BaseBoxes`): BaseBoxes with shape of (m, box_dim)
                or empty.
            boxes2 (:obj:`BaseBoxes`): BaseBoxes with shape of (n, box_dim)
                or empty.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground). Defaults to "iou".
            is_aligned (bool): If True, then m and n must be equal. Defaults
                to False.
            eps (float): A value added to the denominator for numerical
                stability. Defaults to 1e-6.

        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
        """
        boxes1 = boxes1.convert_to('hbox')
        boxes2 = boxes2.convert_to('hbox')
        return bbox_overlaps(
            boxes1.tensor,
            boxes2.tensor,
            mode=mode,
            is_aligned=is_aligned,
            eps=eps)

    @staticmethod
    def from_instance_masks(masks: MaskType) -> 'HorizontalBoxes':
        """Create horizontal boxes from instance masks.

        Args:
            masks (:obj:`BitmapMasks` or :obj:`PolygonMasks`): BitmapMasks or
                PolygonMasks instance with length of n.

        Returns:
            :obj:`HorizontalBoxes`: Converted boxes with shape of (n, 4).
        """
        num_masks = len(masks)
        boxes = np.zeros((num_masks, 4), dtype=np.float32)
        if isinstance(masks, BitmapMasks):
            x_any = masks.masks.any(axis=1)
            y_any = masks.masks.any(axis=2)
            for idx in range(num_masks):
                x = np.where(x_any[idx, :])[0]
                y = np.where(y_any[idx, :])[0]
                if len(x) > 0 and len(y) > 0:
                    # use +1 for x_max and y_max so that the right and bottom
                    # boundary of instance masks are fully included by the box
                    boxes[idx, :] = np.array(
                        [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)
        elif isinstance(masks, PolygonMasks):
            for idx, poly_per_obj in enumerate(masks.masks):
                # simply use a number that is big enough for comparison with
                # coordinates
                xy_min = np.array([masks.width * 2, masks.height * 2],
                                  dtype=np.float32)
                xy_max = np.zeros(2, dtype=np.float32)
                for p in poly_per_obj:
                    xy = np.array(p).reshape(-1, 2).astype(np.float32)
                    xy_min = np.minimum(xy_min, np.min(xy, axis=0))
                    xy_max = np.maximum(xy_max, np.max(xy, axis=0))
                boxes[idx, :2] = xy_min
                boxes[idx, 2:] = xy_max
        else:
            raise TypeError(
                '`masks` must be `BitmapMasks`  or `PolygonMasks`, '
                f'but got {type(masks)}.')
        return HorizontalBoxes(boxes)

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Clip boxes according to the image shape in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
        """
        boxes = self.tensor
        boxes[..., 0] = boxes[..., 0].clamp(0, img_shape[1])
        boxes[..., 1] = boxes[..., 1].clamp(0, img_shape[0])

    def is_inside(self,
                  img_shape: Tuple[int, int],
                  all_inside: bool = False,
                  allowed_border: int = 0) -> BoolTensor:
        """Find boxes inside the image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            all_inside (bool): Whether the boxes are all inside the image or
                part inside the image. Defaults to False.
            allowed_border (int): Boxes that extend beyond the image shape
                boundary by more than ``allowed_border`` are considered
                "outside" Defaults to 0.
        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside
            the image. Assuming the original boxes have shape (m, n, 4),
            the output has shape (m, n).
        """
        img_h, img_w = img_shape
        boxes = self.tensor
        if all_inside:
            pdb.set_trace()
            return (boxes[:, 0] >= -allowed_border) & \
                (boxes[:, 1] >= -allowed_border) & \
                (boxes[:, 2] < img_w + allowed_border) & \
                (boxes[:, 3] < img_h + allowed_border)
        else:
            return (boxes[..., 0] < img_w + allowed_border) & \
                (boxes[..., 1] < img_h + allowed_border) & \
                (boxes[..., 0] > -allowed_border) & \
                (boxes[..., 1] > -allowed_border)


