#!/bin/bash

OUTPUT_FOLDER="/work/Agent_benchmark/slide_seek/post_eval/TCGA_LUNG_Classification/ROI"

python eval_roi_overlap.py "/work/Agent_benchmark/slide_seek/tcga_lusc_subtype" "/work/Agent_benchmark/slide_seek/tcga_luad_subtype" \
       --gt-dir /work/data/TCGA-LUAD-FS/Tumor_annotation/LUAD \
                /work/data/TCGA-LUSC-FS/Tumor_annotation/LUSC \
       --output "${OUTPUT_FOLDER}/roi_overlap_results.csv" \
       --summary-output "${OUTPUT_FOLDER}/roi_overlap_summary.csv" \
       --qa-folder /work/Agent_benchmark/slide_seek/tcga_lusc_subtype /work/Agent_benchmark/slide_seek/tcga_luad_subtype \
       --mrfh-threshold-output "${OUTPUT_FOLDER}/mrfh_by_threshold.csv" \
       --hit-rate-output "${OUTPUT_FOLDER}/hit_rate_by_threshold.csv"


OUTPUT_FOLDER="/work/Agent_benchmark/slide_seek/post_eval/TCGA_BRCA_subtype/ROI"

python eval_roi_overlap.py /work/Agent_benchmark/slide_seek/tcga_brca_subtype \
       --gt-dir /work/data/TCGA-BRCA-FS/Tumor_annotation/BRCA \
       --output "${OUTPUT_FOLDER}/roi_overlap_results.csv" \
       --summary-output "${OUTPUT_FOLDER}/roi_overlap_summary.csv" \
       --qa-folder /work/Agent_benchmark/slide_seek/tcga_brca_subtype \
       --mrfh-threshold-output "${OUTPUT_FOLDER}/mrfh_by_threshold.csv" \
       --hit-rate-output "${OUTPUT_FOLDER}/hit_rate_by_threshold.csv"


OUTPUT_FOLDER="/work/Agent_benchmark/slide_seek/post_eval/CAMELYON16_detection/ROI"

python eval_roi_overlap.py /work/Agent_benchmark/slide_seek/qa_results \
       --gt-dir /work/Agent_dataset/CAMELYON16/csv_output \
       --output "${OUTPUT_FOLDER}/roi_overlap_results.csv" \
       --summary-output "${OUTPUT_FOLDER}/roi_overlap_summary.csv" \
       --qa-folder /work/Agent_benchmark/slide_seek/qa_results \
       --mrfh-threshold-output "${OUTPUT_FOLDER}/mrfh_by_threshold.csv" \
       --hit-rate-output "${OUTPUT_FOLDER}/hit_rate_by_threshold.csv"