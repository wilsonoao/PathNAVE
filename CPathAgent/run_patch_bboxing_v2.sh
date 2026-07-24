#!/bin/bash

python save_patch_bboxes_v2.py /work/nas/WSI/04_LUAD \
    /work/project/Agent_benchmark/CPathAgent/output/test_tcga_luad/TCGA_LUAD  \
    patch_bbox/TCGA_LUAD \
    --patch-size 512 \
    --overlap 0.05 \
    --thumbnail-scale 32


# python save_patch_bboxes_v2.py /data/svs_root /data/reviewed_cases /output/bboxes
