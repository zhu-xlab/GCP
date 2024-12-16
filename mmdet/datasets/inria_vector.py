# Copyright (c) OpenMMLab. All rights reserved.
import pdb
from mmdet.registry import DATASETS
from .coco import CocoDataset
from typing import List, Union
import glob


@DATASETS.register_module()
class InriaVectorDataset(CocoDataset):
    """Dataset for iSAID instance segmentation.

    iSAID: A Large-scale Dataset for Instance Segmentation
    in Aerial Images.

    For more detail, please refer to "projects/iSAID/README.md"
    """
    METAINFO = dict(
        classes=('building'),
        palette=[(0, 0, 255)])

    # def __init__(self, **kwargs):
    #     self.test_mode = test_mode
    def load_data_list(self) -> List[dict]:
        if not self.test_mode:
            return super(InriaVectorDataset, self).load_data_list()

        img_dir = self.data_prefix['img']
        data_list = glob.glob(img_dir + '/*')
        data_list = [{'img_path': x} for x in data_list]
        self.test_data_list = data_list

        return data_list

    # def __len__(self):
    #     if not self.test_mode:
    #         return super(InriaVectorDataset, self).__len__()

    #     return len(self.test_data_list)

