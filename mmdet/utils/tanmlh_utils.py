import numpy as np
from tqdm import tqdm
import time
import torch
import pdb
import math
import cv2
from mmengine.structures import BaseDataElement, InstanceData, PixelData
import torch.nn.functional as F
from mmdet.models.layers.bbox_nms import fast_nms
from mmdet.structures.bbox import bbox_overlaps, obb2xyxy
import mmdet.utils.tanmlh_polygon_utils as polygon_utils
from scipy.sparse.csgraph import connected_components
import shapely

def compute_overlap_matrix(boxes1, boxes2, mode='numpy'):

    if mode == 'numpy':
        # Reshape boxes1 and boxes2 to enable broadcasting
        # boxes1 shape: (N, 1, 4)
        # boxes2 shape: (1, M, 4)
        boxes1 = boxes1[:, np.newaxis, :]
        boxes2 = boxes2[np.newaxis, :, :]

        # Compute conditions for overlapping
        # Overlap along x-axis
        x_overlap = np.logical_not(
            (boxes1[..., 2] < boxes2[..., 0]) | (boxes1[..., 0] > boxes2[..., 2])
        )

        # Overlap along y-axis
        y_overlap = np.logical_not(
            (boxes1[..., 3] < boxes2[..., 1]) | (boxes1[..., 1] > boxes2[..., 3])
        )

        # A matrix where True represents an overlap
        overlap_matrix = np.logical_and(x_overlap, y_overlap)
    elif mode == 'torch':
        # Reshape boxes1 and boxes2 to enable broadcasting
        # boxes1 shape: (N, 1, 4)
        # boxes2 shape: (1, M, 4)
        boxes1 = boxes1[:, None, :]
        boxes2 = boxes2[None, :, :]

        # Compute conditions for overlapping
        # Overlap along x-axis
        x_overlap = torch.logical_not(
            (boxes1[..., 2] < boxes2[..., 0]) | (boxes1[..., 0] > boxes2[..., 2])
        )

        # Overlap along y-axis
        y_overlap = torch.logical_not(
            (boxes1[..., 3] < boxes2[..., 1]) | (boxes1[..., 1] > boxes2[..., 3])
        )

        # A matrix where True represents an overlap
        overlap_matrix = torch.logical_and(x_overlap, y_overlap)

    else:
        raise ValueError()

    return overlap_matrix

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])  # Path compression
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            # Union by rank
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1

def do_boxes_overlap(box1, box2):
    # Check if two boxes overlap
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

def assign_boxes_to_grids_v2(boxes, grid_size, bounds):
    # Calculate the grid indices for each bounding box
    grid_indices_x = np.floor((boxes[:, 0] - bounds[0]) / grid_size[0]).astype(int)
    grid_indices_y = np.floor((boxes[:, 1] - bounds[1]) / grid_size[1]).astype(int)
    
    # Create a dictionary to hold the bounding boxes in each grid
    grid_dict = {}
    for idx, (gx, gy) in enumerate(zip(grid_indices_x, grid_indices_y)):
        grid_key = (gx, gy)
        if grid_key not in grid_dict:
            grid_dict[grid_key] = []
        grid_dict[grid_key].append((idx, boxes[idx]))

    return grid_dict


def find_connected_components(boxes, grid_size):
    # Determine bounds based on the input boxes
    min_x = np.min(boxes[:, 0])
    min_y = np.min(boxes[:, 1])
    max_x = np.max(boxes[:, 2])
    max_y = np.max(boxes[:, 3])

    # Initialize Union-Find
    uf = UnionFind(len(boxes))

    # Assign boxes to grids
    grid_dict = assign_boxes_to_grids(boxes, grid_size, [min_x, min_y, max_x, max_y])

    # Find overlaps within each grid
    for grid_boxes in grid_dict.values():
        for i in range(len(grid_boxes)):
            for j in range(i + 1, len(grid_boxes)):
                if do_boxes_overlap(grid_boxes[i][1], grid_boxes[j][1]):
                    uf.union(grid_boxes[i][0], grid_boxes[j][0])

    return uf

def find_connected_boxes(boxes, grid_size):

    uf = find_connected_components(boxes, grid_size)

    # Get components (post-process to get each component)
    components = {}
    for i in range(len(boxes)):
        root = uf.find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    return components

def create_grid(x_min, y_min, x_max, y_max, h, w):
    """
    Create a grid of non-overlapping cells that exactly cover the bounding box.

    Parameters:
        x_min (float): Minimum x-coordinate of the bounding box.
        y_min (float): Minimum y-coordinate of the bounding box.
        x_max (float): Maximum x-coordinate of the bounding box.
        y_max (float): Maximum y-coordinate of the bounding box.
        h (float): Desired maximum height of each grid cell.
        w (float): Desired maximum width of each grid cell.

    Returns:
        np.ndarray: Array of grid cells with shape (N, 4), 
                    where each row is [x_min_cell, y_min_cell, x_max_cell, y_max_cell].
    """
    # Compute the total width and height of the bounding box
    total_width = x_max - x_min
    total_height = y_max - y_min

    # Adjust grid size to exactly fit the bounding box
    num_x = int(np.ceil(total_width / w))  # Number of grid cells in x direction
    num_y = int(np.ceil(total_height / h))  # Number of grid cells in y direction

    adjusted_w = total_width / num_x  # Adjusted width to fit exactly
    adjusted_h = total_height / num_y  # Adjusted height to fit exactly

    grid_cells = []

    # Generate grid cells
    for i in range(num_x):
        for j in range(num_y):
            # Calculate grid cell coordinates
            cell_x_min = x_min + i * adjusted_w
            cell_y_min = y_min + j * adjusted_h
            cell_x_max = cell_x_min + adjusted_w
            cell_y_max = cell_y_min + adjusted_h

            grid_cells.append([cell_x_min, cell_y_min, cell_x_max, cell_y_max])

    return np.array(grid_cells)

def get_crop_boxes(img_H, img_W, crop_size=(256, 256), stride=(192, 192)):
    # prepare locations to crop

    num_rows = math.ceil((img_H - crop_size[0]) / stride[0]) if \
        math.ceil((img_H - crop_size[0]) / stride[0]) * stride[0] + crop_size[0] >= img_H \
        else math.ceil( (img_H - crop_size[0]) / stride[0]) + 1

    num_cols = math.ceil((img_W - crop_size[1]) / stride[1]) if math.ceil(
        (img_W - crop_size[1]) /
        stride[1]) * stride[1] + crop_size[1] >= img_W else math.ceil(
            (img_W - crop_size[1]) / stride[1]) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * stride[1]
    ymin = y * stride[0]

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + crop_size[1] > img_W, img_W - xmin - crop_size[1],
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + crop_size[0] > img_H, img_H - ymin - crop_size[0],
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + crop_size[1], img_W),
        np.minimum(ymin + crop_size[0], img_H)
    ], axis=1)

    return boxes

def get_patch_weight(patch_size):
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

def mosaic_instance_data(instance_list, offsets, mask_shape=None, pad_shape=None, mask_up_scale=1.0, device='cpu'):
    assert len(instance_list) == len(offsets)
    offsets = offsets.repeat(1,2)
    merged_instance = InstanceData()

    if 'scores' in instance_list[0]:
        new_scores = []

    if 'bboxes' in instance_list[0]:
        new_bboxes = []

    if 'masks' in instance_list[0]:
        new_masks = []

    if 'labels' in instance_list[0]:
        new_labels = []

    if 'segmentations' in instance_list[0]:
        new_polygons = []

    if 'sem_seg' in instance_list[0]:
        C = instance_list[0].sem_seg.shape[1]
        new_sem_seg = instance_list[0].sem_seg.new_zeros(C, *mask_shape)
        cnt_sem_seg = instance_list[0].sem_seg.new_zeros(C, *mask_shape)

    for i, instance in enumerate(instance_list):
        if 'scores' in instance:
            new_scores.append(instance['scores'])

        if 'bboxes' in instance:
            if instance['bboxes'].shape[1] == 4:
                if pad_shape is not None:
                    new_bboxes.append(instance['bboxes'] * mask_up_scale + offsets[i] - (pad_shape[0] - mask_shape[0]) // 2)
                else:
                    new_bboxes.append(instance['bboxes'] * mask_up_scale + offsets[i])
            elif instance['bboxes'].shape[1] == 5:
                if pad_shape is not None:
                    bboxes = instance['bboxes'].clone()
                    bboxes[:,:4] *= mask_up_scale
                    bboxes[:,:2] += offsets[i, :2] - (pad_shape[0] - mask_shape[0]) // 2
                    new_bboxes.append(bboxes)
                else:
                    bboxes = instance['bboxes'].clone()
                    bboxes[:,:4] *= mask_up_scale
                    bboxes[:,:2] += offsets[i, :2]
                    new_bboxes.append(bboxes)

        if 'sem_seg' in instance:
            N, C, h, w = instance.sem_seg.shape
            start_x, start_y = offsets[i, :2]
            start_x2 = 0 if start_x >= 0 else -start_x
            start_y2 = 0 if start_y >= 0 else -start_y
            end_x2 = w if start_x + w <= mask_shape[1] else mask_shape[1] - start_x
            end_y2 = h if start_y + h <= mask_shape[0] else mask_shape[0] - start_y

            weights = torch.tensor(get_patch_weight(h), device=instance.sem_seg.device)
            weighted_sem_seg = instance.sem_seg * weights.view(N,1,h,w)
            new_sem_seg[:, max(start_y, 0):start_y+h, max(start_x, 0):start_x+w] += \
                    weighted_sem_seg[0, :, start_y2:end_y2, start_x2:end_x2]
            cnt_sem_seg[:, max(start_y, 0):start_y+h, max(start_x, 0):start_x+w] += 1


        if 'masks' in instance:
            if mask_shape is not None:
                if pad_shape is not None:
                    new_mask = instance['masks'].new_zeros(len(instance), *pad_shape)
                else:
                    new_mask = instance['masks'].new_zeros(len(instance), *mask_shape)

                N, H, W = instance['masks'].shape
                H = int(H * mask_up_scale)
                W = int(W * mask_up_scale)

                cur_mask = F.interpolate(instance['masks'].unsqueeze(1).float(), size=(H, W), mode='bilinear')[:,0].bool()
                new_mask[:, offsets[i][1]:offsets[i][1]+H, offsets[i][0]:offsets[i][0]+W] = cur_mask
                pad_offset = (pad_shape[0] - mask_shape[0]) // 2
                temp = new_mask[:, pad_offset:pad_shape[0]-pad_offset, pad_offset:pad_shape[1]-pad_offset]
                new_masks.append(temp)
            else:
                new_masks.append(instance['masks'])

        if 'labels' in instance:
            new_labels.append(instance['labels'])

        if 'segmentations' in instance:
            poly_jsons = instance['segmentations']
            new_poly_jsons = []
            for j, poly_json in enumerate(poly_jsons):
                # if poly_json['type'] == 'Polygon':
                #     rings = poly_json['coordinates']
                #     new_rings = []
                #     for ring in rings:
                #         temp = np.array(ring) + offsets[i:i+1, :2].cpu().numpy()
                #         new_rings.append(temp.tolist())

                #     temp = dict(
                #         type='Polygon',
                #         coordinates=new_rings
                #     )
                #     new_poly_jsons.append(temp)
                # elif poly_json['type'] == 'MultiPolygon':
                new_poly_json = polygon_utils.transform_polygon(
                    poly_json, offsets=offsets[i, :2].cpu().tolist()
                )
                new_poly_jsons.append(new_poly_json)

            new_polygons.extend(new_poly_jsons)

    if 'scores' in instance_list[0]:
        merged_instance.scores = torch.cat(new_scores).to(device)

    if 'bboxes' in instance_list[0]:
        merged_instance.bboxes = torch.cat(new_bboxes).to(device)

    if 'masks' in instance_list[0]:
        merged_instance.masks = torch.cat(new_masks).to(device)

    if 'labels' in instance_list[0]:
        merged_instance.labels = torch.cat(new_labels).to(device)

    if 'segmentations' in instance_list[0]:
        merged_instance.segmentations = new_polygons

    if 'sem_seg' in instance_list[0]:
        merged_instance.sem_seg = (new_sem_seg / (cnt_sem_seg + 1e-8))[None]

    return merged_instance

def sample_instances(instance, num_sample=-1):
    N = len(instance)
    if num_sample <= 0 or N <= num_sample:
        return instance

    idxes = np.random.permutation(N)[:num_sample]
    return instance[idxes]

def instance_nms(instance, nms_cfg, eps=1e-8, verbose=False):

    idx = torch.argsort(instance.scores, descending=True)
    instance = instance[idx]
    scores = instance.scores
    bboxes = instance.bboxes

    scores = torch.stack([scores, 1-scores], dim=1)

    score_thr = nms_cfg.get('score_thr', 0.0)
    iou_thr = nms_cfg.get('iou_thr', 0.7)
    # instance = instance[instance.scores >= score_thr]
    nms_type = nms_cfg.get('nms_type', 'bbox')

    if nms_type == 'bbox':
        _, _, _, idx = fast_nms(bboxes, scores, scores[:,0:1], iou_thr=iou_thr, score_thr=score_thr, top_k=len(scores), return_idx=True)
        return instance[idx]

    elif nms_type == 'no_overlap':
        masks = instance.masks
        half_iou_thr1 = 0.8
        half_iou_thr2 = 0.2

        if bboxes.shape[1] == 5:
            bboxes = obb2xyxy(bboxes)[:,:4]

        iou_mat = bbox_overlaps(bboxes, bboxes)
        n_comp, label_comp = connected_components(csgraph=(iou_mat > 0.0).cpu().numpy(), directed=False)
        result_idxes = []
        for i in range(n_comp):
            cur_idxes = (label_comp == i).nonzero()[0]
            if len(cur_idxes) == 1:
                result_idxes.append(cur_idxes.item())
                continue

            union_masks = masks[cur_idxes].any(dim=0)
            ori_area = union_masks.sum()
            for j in cur_idxes:
                intersect = masks[j] * union_masks
                half_iou1 = intersect.sum() / (masks[j].sum() + eps)
                # half_iou2 = intersect.sum() / (ori_area + eps)

                if half_iou1 >= half_iou_thr1:
                # if half_iou1 >= half_iou_thr1 or half_iou2 >= half_iou_thr2:
                    masks[j] = masks[j] * intersect
                    union_masks = union_masks ^ intersect
                    result_idxes.append(j)

        return instance[result_idxes] if len(result_idxes) > 0 else instance[:0]

    elif nms_type == 'polygon':

        if verbose:
            tic = time.time()

        def cal_half_iou(polygon1, polygon2, eps=1e-8):
            intersection = polygon1.intersection(polygon2)
            # union = polygon1.union(polygon2)
            iou = intersection.area / (polygon1.area + eps)
            return iou

        grid_size = nms_cfg.get('grid_size', (1024, 1024))
        half_iou_thr1 = nms_cfg.get('half_iou_thr1', 0.1)
        half_iou_thr2 = nms_cfg.get('half_iou_thr2', 0.5)

        poly_shps = [shapely.geometry.shape(x) for x in instance.segmentations]
        boxes = np.array([x.bounds for x in poly_shps])

        if len(boxes) == 0:
            return instance[:0]

        bounds = np.concatenate([boxes.min(axis=0)[:2], boxes.max(axis=0)[2:]])

        grids = create_grid(bounds[0], bounds[1], bounds[2], bounds[3], grid_size[0], grid_size[1])
        # grids = get_crop_boxes(
        #     bounds[3] - bounds[1], bounds[2] - bounds[0], grid_size, grid_size
        # )
        # # grids = np.concatenate([bounds[:2], bounds[:2]]).reshape(1,-1) + grids
        overlap_mat = compute_overlap_matrix(grids, boxes)

        result_idxes = []
        remove_idxes = []
        for i in range(len(grids)):
            cur_box_idxes = overlap_mat[i].nonzero()[0]
            cur_boxes = torch.tensor(boxes[cur_box_idxes])
            iou_mat = bbox_overlaps(cur_boxes, cur_boxes)
            # iou_mat = compute_overlap_matrix(cur_boxes, cur_boxes)

            n_comp, label_comp = connected_components(csgraph=(iou_mat > 0.0).cpu().numpy(), directed=False)
            for j in range(n_comp):
                cur_idxes = (label_comp == j).nonzero()[0]
                cur_poly_shps = [poly_shps[cur_box_idxes[x]] for x in cur_idxes]

                # if len(cur_idxes) == 1:
                #     result_idxes.append(cur_box_idxes[cur_idxes.item()])
                #     continue
                # all_union = shapely.ops.unary_union(cur_poly_shps)
                cur_union = shapely.geometry.Polygon()
                for k, poly_shp in enumerate(cur_poly_shps):
                    if not poly_shp.is_valid:
                        remove_idxes.append(cur_box_idxes[cur_idxes[k]])
                        continue

                    iou1 = cal_half_iou(poly_shp, cur_union)
                    if iou1 <= half_iou_thr1:
                        # and iou2 >= half_iou_thr2:
                        result_idxes.append(cur_box_idxes[cur_idxes[k]])
                        # cur_union = shapely.ops.unary_union([cur_union, poly_shp])
                        cur_union = cur_union.union(poly_shp)
                        # union = union.difference(poly_shp)
                        # if union.geom_type == 'GeometryCollection':
                        #     union = shapely.GeometryCollection([geom for geom in union.geoms if not isinstance(geom, shapely.LineString)])
                    else:
                        remove_idxes.append(cur_box_idxes[cur_idxes[k]])

        remove_idxes = set(remove_idxes)
        result_idxes = list(set(list(range(len(poly_shps)))) - remove_idxes)

        if verbose:
            print(f'time elapse of the nms process: {time.time() - tic}')

        # new_nms_cfg = nms_cfg.copy()
        # new_nms_cfg['nms_type'] = 'no_overlap'
        # new_instance = instance_nms(instance, new_nms_cfg, eps=1e-8, verbose=True)
        # if len(new_instance) != len(result_idxes):
        #     pdb.set_trace()

        return instance[result_idxes] if len(result_idxes) > 0 else instance[:0]

    elif nms_type == 'none':
        return instance

def iou_binary_mask(mask1, mask2):
    return (mask1 & mask2) / (mask1 | mask2)

def filter_border_instance(instance, border_width, width, filter_type='bbox'):
    if filter_type == 'bbox':
        bboxes = instance.bboxes
        flag_x = (bboxes[:,2] <= border_width) | (bboxes[:,0] >= width - border_width)
        flag_y = (bboxes[:,3] <= border_width) | (bboxes[:,1] >= width - border_width)
        return instance[~(flag_x | flag_y)]
    else:
        pdb.set_trace()

def pad_tensor_to_center(tensor, pad_h, pad_w):
    # Get original dimensions
    B, C, H, W = tensor.shape

    # Calculate padding for height and width
    pad_top = (pad_h - H) // 2
    pad_bottom = pad_h - H - pad_top
    pad_left = (pad_w - W) // 2
    pad_right = pad_w - W - pad_left

    # Apply padding (torch.nn.functional.pad takes padding in reverse order: (left, right, top, bottom))
    # padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

    return padded_tensor

def crop_featmap(featmap, crop_boxes, stride):
    # featmap: B, C, H, W

    _,_,H,W = featmap.shape
    cropped_featmaps = []
    crop_boxes = crop_boxes // stride
    for crop_box in crop_boxes:
        x0,y0,x1,y1 = crop_box
        crop_featmap = featmap[:,:,y0:y1,x0:x1]
        cropped_featmaps.append(F.interpolate(crop_featmap, (W, H)))

    return torch.cat(cropped_featmaps, dim=0)

def paste_masks(masks, boxes, H, W):
    # masks: N, H, W
    # boxes: N, 4

    N = len(masks)
    # w_scale = (boxes[:,2] - boxes[:,0]) / H
    # h_scale = (boxes[:,3] - boxes[:,1]) / W

    ws = boxes[:,2] - boxes[:,0]
    hs = boxes[:,3] - boxes[:,1]

    assert (ws == ws[0]).all()
    assert (hs == hs[0]).all()

    # scales = torch.stack([w_scale, h_scale], dim=1).numpy()
    offsets = boxes[:,:2]

    resized_masks = F.interpolate(masks.unsqueeze(1), (ws[0], hs[0])).squeeze(1)

    new_masks = torch.zeros_like(masks)
    for i, mask in enumerate(resized_masks):
        new_masks[i, offsets[i,1]:offsets[i,1]+hs[0], offsets[i,0]:offsets[i,0]+ws[0]] = mask

    return new_masks


def generate_angles(N):

    # Generate initial angles in radians, from 0 to 2*pi (exclusive)
    # initial_angles = torch.linspace(0, 2 * torch.pi, steps=N, endpoint=False)
    initial_angles = torch.linspace(0, 2 * torch.pi, N+1)[:-1]

    # Ensure angles are kept in a matrix of shape (N, 1) to facilitate broadcasting
    initial_angles = initial_angles.unsqueeze(1)

    # Generate a tensor of angular increments: [0, pi/2, pi, 3pi/2] which are perpendicular steps
    increments = torch.tensor([0, torch.pi/2, torch.pi, 3*torch.pi/2])

    # Add increments to initial angles and apply modulo 2*pi to wrap around the circle
    angles_radians = (initial_angles + increments) % (2 * torch.pi)

    return angles_radians

def get_angles(points):
    # points: (N, 2)
    u = points
    v = torch.roll(u, shifts=[-1], dims=[0])
    vec = v - u
    angle = torch.atan2(vec[:,1], vec[:,0])

    return angle

def batch_get_angles(points):
    # points: (B, N, 2)
    u = points
    v = torch.roll(u, shifts=[-1], dims=[1])
    vec = v - u
    angle = torch.atan2(vec[:, :,1], vec[:, :,0])

    return angle

def get_base_angle_idxes(angles, base_angles):
    # angles: (B,N)
    # base_angles: (M, 4)
    B, N = angles.shape
    M, _ = base_angles.shape

    diff = angles.view(B,N,1,1) - base_angles.view(1,1,M,4)
    d1 = (diff.abs() % (torch.pi * 2))
    d2 = 2 * torch.pi - (diff.abs() % (torch.pi * 2))
    min_ang_dis = torch.where(d1 < d2, d1, d2)

    ang_dis = min_ang_dis.min(dim=-1)[0].mean(dim=1)
    ang_idxes = ang_dis.argmin(dim=1)

    return ang_dis, ang_idxes


def poly_overlaps(polys_A, polys_B, grid_size=(1024, 1024), iou_type='iou', debug=False):

    def cal_iou(polygon1, polygon2, eps=1e-8):
        intersection = polygon1.intersection(polygon2)
        union = polygon1.union(polygon2)
        iou = intersection.area / (union.area + eps)
        return iou

    def cal_half_iou(polygon1, polygon2, eps=1e-8):
        intersection = polygon1.intersection(polygon2)
        # union = polygon1.union(polygon2)
        iou = intersection.area / (polygon1.area + eps)
        return iou

    if iou_type == 'iou':
        iou_fun = cal_iou
    elif iou_type == 'half_iou':
        iou_fun = cal_half_iou

    poly_shps_A = polys_A.get_shapely()
    poly_shps_B = polys_B.get_shapely()
    iou_mat = np.zeros((len(polys_A), len(polys_B)))

    if len(polys_A) == 0 or len(polys_B) == 0:
        return iou_mat

    bounds_A = polys_A.get_bounds()
    bounds_B = polys_B.get_bounds()
    bounds_min = np.concatenate([bounds_A, bounds_B]).min(axis=0)[:2]
    bounds_max = np.concatenate([bounds_A, bounds_B]).max(axis=0)[2:]
    bounds = np.concatenate([bounds_min, bounds_max])

    if grid_size is not None:
        grids = create_grid(bounds[0], bounds[1], bounds[2], bounds[3], grid_size[0], grid_size[1])
        # grids = np.concatenate([bounds[:2], bounds[:2]]).reshape(1,-1) + grids
    else:
        grids = bounds[None]

    overlap_mat_A = compute_overlap_matrix(grids, bounds_A)
    overlap_mat_B = compute_overlap_matrix(grids, bounds_B)

    result_idxes = []
    A_idxes = []
    B_idxes = []
    if debug:
        pdb.set_trace()
    for i in range(len(grids)):
        cur_A_idxes = overlap_mat_A[i].nonzero()[0]
        cur_B_idxes = overlap_mat_B[i].nonzero()[0]
        cur_overlap_mat = compute_overlap_matrix(bounds_A[cur_A_idxes], bounds_B[cur_B_idxes])
        if cur_overlap_mat.sum() == 0:
            continue

        row_idxes, col_idxes = cur_overlap_mat.nonzero()
        for (row_id, col_id) in zip(row_idxes, col_idxes):
            cur_iou = iou_fun(poly_shps_A[cur_A_idxes[row_id]], poly_shps_B[cur_B_idxes[col_id]])
            iou_mat[cur_A_idxes[row_id], cur_B_idxes[col_id]] = cur_iou

    return iou_mat

def generate_instances_by_polygon_masks(polygon_masks, entries=[]):
    pdb.set_trace()
