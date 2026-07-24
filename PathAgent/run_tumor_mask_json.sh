#!/bin/bash 

python mask_tumor_region.py \
    --feature_root /work/Agent_benchmark/PathAgent/result/CAMELYON16_self_correction/img_features \
    --xml_root /work/Agent_dataset/CAMELYON16/annotations \
    --use_intersection \
    --output_root /work/Agent_benchmark/PathAgent/result/CAMELYON16_self_correction/mask_tumor