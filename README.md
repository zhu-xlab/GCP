Official Implementation of "Global Collinearity-aware Polygonizer for Polygonal Building Mapping in Remote Sensing"
To re-implement the model in the paper, please follow the bellow steps.

## Datasets Preparation
1. Crowd AI dataset
   Download the dataset at https://www.aicrowd.com/challenges/mapping-challenge/dataset_files
   
   Unzip train.trg.gz and val.trg.gz, and place them in your data folder.
   
   Modify the configuration file in config/_base_/crowd_ai_bs16.py. Change "data_root" attribute in line 3 to your data folder.
   
   Make sure that in train, val and test dataloader settings in line 62, 79 and 96, data_root + ann_file and correctly locate your annotation file, and data_root + data_prefix and locate your image folder.
   
3. Whu-Mix-Vector dataset
   Download the dataset at

   Unzip and place the dataset in your data folder, and modify the configuration file in config/_base_/whu_mix_vector accordingly just like CrowdAI dataset.

## Model weights
