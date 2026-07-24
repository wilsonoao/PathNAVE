#!/bin/bash

OUTPUT_FOLDER="/work/Agent_benchmark/CPathAgent/post_eval/TCGA_LUNG_Classification/ROI"

python eval_roi_overlap.py "/work/Agent_benchmark/CPathAgent/output/test_tcga_lusc/TCGA_LUSC" "/work/Agent_benchmark/CPathAgent/output/test_tcga_luad/TCGA_LUAD"\
       --gt-dir /work/data/TCGA-LUAD-FS/Tumor_annotation/LUAD \
                /work/data/TCGA-LUSC-FS/Tumor_annotation/LUSC \
       --output "${OUTPUT_FOLDER}/roi_overlap_results.csv" \
       --summary-output "${OUTPUT_FOLDER}/roi_overlap_summary.csv" \
       --qa-folder /work/Agent_benchmark/CPathAgent/output/test_tcga_lusc/TCGA_LUSC /work/Agent_benchmark/CPathAgent/output/test_tcga_luad/TCGA_LUAD \
       --patch-size 512 \
       --ids-bbox /work/Agent_benchmark/CPathAgent/patch_bbox/TCGA_LUSC \
       --mrfh-threshold-output "${OUTPUT_FOLDER}/mrfh_by_threshold.csv" \
       --hit-rate-output "${OUTPUT_FOLDER}/hit_rate_by_threshold.csv"



OUTPUT_FOLDER="/work/Agent_benchmark/CPathAgent/post_eval/TCGA_BRCA_subtype/ROI"

python eval_roi_overlap.py /work/Agent_benchmark/CPathAgent/output/test_tcga_brca/TCGA_BRCA \
       --gt-dir /work/data/TCGA-BRCA-FS/Tumor_annotation/BRCA \
       --output "${OUTPUT_FOLDER}/roi_overlap_results.csv" \
       --summary-output "${OUTPUT_FOLDER}/roi_overlap_summary.csv" \
       --qa-folder /work/Agent_benchmark/CPathAgent/output/test_tcga_brca/TCGA_BRCA \
       --patch-size 512 \
       --ids-bbox /work/Agent_benchmark/CPathAgent/patch_bbox/TCGA_BRCA \
       --mrfh-threshold-output "${OUTPUT_FOLDER}/mrfh_by_threshold.csv" \
       --hit-rate-output "${OUTPUT_FOLDER}/hit_rate_by_threshold.csv"




OUTPUT_FOLDER="/work/Agent_benchmark/CPathAgent/post_eval/CAMELYON16_detection/ROI"

python eval_roi_overlap.py /work/Agent_benchmark/CPathAgent/output/CAMELYON16 \
       --gt-dir /work/Agent_dataset/CAMELYON16/csv_output \
       --output "${OUTPUT_FOLDER}/roi_overlap_results.csv" \
       --summary-output "${OUTPUT_FOLDER}/roi_overlap_summary.csv" \
       --qa-folder /work/Agent_benchmark/CPathAgent/output/CAMELYON16 \
       --patch-size 512 \
       --ids-bbox /work/Agent_benchmark/CPathAgent/patch_bbox/CAMELYON16 \
       --mrfh-threshold-output "${OUTPUT_FOLDER}/mrfh_by_threshold.csv" \
       --hit-rate-output "${OUTPUT_FOLDER}/hit_rate_by_threshold.csv"
