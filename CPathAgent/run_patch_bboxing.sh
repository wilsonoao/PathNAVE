#!/bin/bash

python save_patch_bboxes.py /work/Agent_dataset/CAMELYON16/slides \
    patch_bbox/CAMELYON16 \
    --patch-size 512 \
    --overlap 0.05 \
    --thumbnail-scale 32