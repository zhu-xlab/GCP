import json
import pdb
import random

in_path = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_sample_europe2/coco_ann/global_quartely_2023q2.json'
out_path = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_sample_europe2/coco_ann/small-global_quartely_2023q2.json'
coco_dict = json.load(open(in_path, 'r'))

num_chosen = 100
images = coco_dict['images']

chosen_images = random.choices(images, k=num_chosen)
selected_img_ids = {img['id']:True for img in chosen_images}

selected_anns = []
for ann in coco_dict['annotations']:
    if ann['image_id'] in selected_img_ids:
        selected_anns.append(ann)

new_coco_dict = dict(
    images=chosen_images,
    annotations=selected_anns,
    categories=coco_dict['categories']
)

new_coco_str = json.dumps(new_coco_dict)

with open(out_path, 'w') as f:
    f.write(new_coco_str)
