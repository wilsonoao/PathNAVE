#!/bin/bash

python ROI.py \
       --result_root /work/Agent_benchmark/Pathology-CoT/result/CAMELYON16 \
       --annotation_dir /work/Agent_dataset/CAMELYON16/annotations \
       --slide_dir /work/Agent_dataset/CAMELYON16/slides \
       --output ./test.json \

