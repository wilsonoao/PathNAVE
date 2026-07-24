#!/bin/bash

python coordinate_generation.py \
  --source /work/Agent_dataset/CAMELYON16/slides  \
  --save_dir /work/Agent_benchmark/PathAgent/result/CAMELYON16 \
  --preset tcga.csv \
  --step_size 4096 \
  --patch_size 4096 \
  --patch \
  --seg \