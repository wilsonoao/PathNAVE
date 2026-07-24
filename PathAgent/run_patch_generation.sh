#!/bin/bash

python data_preparation_script/patch_generation.py \
  --h5_dir /work/Agent_benchmark/PathAgent/result/CAMELYON16/patches \
  --slide_dir /work/Agent_dataset/CAMELYON16/slides \
  --output_root /work/Agent_benchmark/PathAgent/result/CAMELYON16/patches_output \
  --patch_size 4096