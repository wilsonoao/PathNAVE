#!/bin/bash

# python data_preparation_script/split_files.py \
#   --image_dir /work/Agent_dataset/Slidebench_VQA_BCNB/patches \
#   --save_dir /work/Agent_benchmark/PathAgent/result/Slidebench_bcnb/split_name \
#   --num_splits 2

# python data_preparation_script/split_files.py \
#   --image_dir /work/Agent_benchmark/PathAgent/result/TCGA_LUSC/patches_output \
#   --save_dir /work/Agent_benchmark/PathAgent/result/TCGA_LUSC/split_name \
#   --num_splits 2

python data_preparation_script/split_files.py \
  --image_dir /work/Agent_benchmark/PathAgent/result/CAMELYON16/patches_output \
  --save_dir /work/Agent_benchmark/PathAgent/result/CAMELYON16/split_name \
  --num_splits 2