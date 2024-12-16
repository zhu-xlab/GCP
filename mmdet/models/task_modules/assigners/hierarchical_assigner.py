import torch
from typing import Optional, Union
from .base_assigner import BaseAssigner
from .assign_result import AssignResult
import pdb

from mmengine.structures import InstanceData
from mmdet.registry import TASK_UTILS
from mmdet.utils import tanmlh_utils

@TASK_UTILS.register_module()
class HierarchicalAssigner(BaseAssigner):
    def __init__(self, base_assigner, img_size, grid_size):
        self.base_assigner = TASK_UTILS.build(base_assigner)
        self.img_size = img_size
        self.grid_size = grid_size

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:

        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes
        else:
            gt_bboxes_ignore = None

        assigned_gt_inds = torch.zeros(len(priors), dtype=torch.long, device=priors.device)
        max_overlaps = torch.zeros(len(priors), device=priors.device) - 1
        assigned_labels = torch.zeros(len(priors), dtype=torch.long, device=priors.device)
        num_gts = len(gt_labels)


        import time
        tic = time.time()

        grids = tanmlh_utils.get_crop_boxes(
            self.img_size[0], self.img_size[1],
            self.grid_size, self.grid_size
        )
        grids = torch.tensor(grids, device=gt_bboxes.device)
        grid_prior_overlaps = tanmlh_utils.compute_overlap_matrix(grids, priors, mode='torch')
        grid_gt_overlaps = tanmlh_utils.compute_overlap_matrix(grids, gt_bboxes, mode='torch')


        for grid_id in range(len(grids)):
            cur_prior_idxes = grid_prior_overlaps[grid_id]
            cur_gt_idxes = grid_gt_overlaps[grid_id].nonzero().flatten()

            cur_pred_instances = InstanceData(priors=priors[cur_prior_idxes])
            cur_gt_instances = InstanceData(
                bboxes=gt_bboxes[cur_gt_idxes],
                labels=gt_labels[cur_gt_idxes]
            )
            cur_assign_result = self.base_assigner.assign(cur_pred_instances, cur_gt_instances, gt_instances_ignore)

            cur_assigned_gt_inds = torch.where(cur_assign_result.gt_inds == -1, -1, 0)
            pos_idxes = (cur_assign_result.gt_inds > 0)
            cur_assigned_gt_inds[pos_idxes] = cur_gt_idxes[cur_assign_result.gt_inds[pos_idxes] - 1] + 1

            assigned_gt_inds[cur_prior_idxes] = torch.where(
                max_overlaps[cur_prior_idxes] < cur_assign_result.max_overlaps,
                cur_assigned_gt_inds, assigned_gt_inds[cur_prior_idxes]
            )
            assigned_labels[cur_prior_idxes] = torch.where(
                max_overlaps[cur_prior_idxes] < cur_assign_result.max_overlaps,
                cur_assign_result.labels, assigned_labels[cur_prior_idxes]
            )
            max_overlaps[cur_prior_idxes] = torch.where(
                max_overlaps[cur_prior_idxes] < cur_assign_result.max_overlaps,
                cur_assign_result.max_overlaps, max_overlaps[cur_prior_idxes]
            )

        # print(f'{time.time() - tic}')
        # print(num_gts)

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels)
