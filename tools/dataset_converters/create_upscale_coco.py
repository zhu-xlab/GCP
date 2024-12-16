import json
import pdb
import random
import numpy as np


# in_path = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_sample_europe2/coco_ann/global_quartely_2023q2.json'
# out_path1 = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_sample_europe2/coco_ann/small-global_quartely_2023q2.json'
# out_path2 = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_sample_europe2/coco_ann/upscale-global_quartely_2023q2.json'

in_path = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_sample_africa/coco_ann/global_quartely_2023q2.json'
out_path1 = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_sample_africa/coco_ann/small-global_quartely_2023q2.json'
out_path2 = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_sample_africa/coco_ann/upscale-global_quartely_2023q2.json'
coco_dict = json.load(open(in_path, 'r'))

num_chosen = 100
upscale = 4.
images = coco_dict['images']

chosen_images = random.choices(images, k=num_chosen)
selected_img_ids = {img['id']:True for img in chosen_images}

small_anns = []
up_anns = []
for ann in coco_dict['annotations']:
    if ann['image_id'] in selected_img_ids:
        new_polygon = []
        for coords in ann['segmentation']:
            new_polygon.append((np.array(coords) * upscale).tolist())

        small_anns.append(ann.copy())
        ann['segmentation'] = new_polygon
        up_anns.append(ann)

new_coco_dict = dict(
    images=chosen_images,
    annotations=small_anns,
    categories=coco_dict['categories']
)

new_coco_str = json.dumps(new_coco_dict)

with open(out_path1, 'w') as f:
    f.write(new_coco_str)

for img in chosen_images:
    img['width'] = int(img['width'] * upscale)
    img['height'] = int(img['height'] * upscale)

new_coco_dict = dict(
    images=chosen_images,
    annotations=up_anns,
    categories=coco_dict['categories']
)

new_coco_str = json.dumps(new_coco_dict)

with open(out_path2, 'w') as f:
    f.write(new_coco_str)
