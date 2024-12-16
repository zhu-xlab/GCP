# Copyright (c) OpenMMLab. All rights reserved.
from .coco_api import COCO, COCOeval, COCOPanoptic
from .cocoeval_mp import COCOevalMP
from .coco_building import COCOeval as COCOevalBuilding

__all__ = ['COCO', 'COCOeval', 'COCOPanoptic', 'COCOevalMP', 'COCOevalBuilding']
