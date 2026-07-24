#!/bin/bash

python fact_matching.py --provider openai --model gpt-4o \
  --atom_dir /work/Agent_benchmark/PathAgent/post_eval/CAMELYON16_detection/Evidence/Atom \
  --output_dir /work/Agent_benchmark/PathAgent/post_eval/CAMELYON16_detection/Evidence/Matching \
  --result_dir /work/Agent_benchmark/PathAgent/result

python fact_matching.py --provider openai --model gpt-4o \
  --atom_dir /work/Agent_benchmark/PathAgent/post_eval/TCGA_BRCA_subtype/Evidence/Atom \
  --output_dir /work/Agent_benchmark/PathAgent/post_eval/TCGA_BRCA_subtype/Evidence/Matching \
  --result_dir /work/Agent_benchmark/PathAgent/result

python fact_matching.py --provider openai --model gpt-4o \
  --atom_dir /work/Agent_benchmark/PathAgent/post_eval/TCGA_LUNG_Classification/Evidence/Atom \
  --output_dir /work/Agent_benchmark/PathAgent/post_eval/TCGA_LUNG_Classification/Evidence/Matching \
  --result_dir /work/Agent_benchmark/PathAgent/result
