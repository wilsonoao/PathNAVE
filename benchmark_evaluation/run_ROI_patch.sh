#!/bin/bash

python ROI_patch.py \
  --result_dir /work/Agent_benchmark/PathAgent/result/CAMELYON16/results \
  --mask_dir /work/Agent_benchmark/PathAgent/result/CAMELYON16_self_correction/mask_tumor \
  --output eval.json \
  --output_summary eval_summary.json \
  --verbose