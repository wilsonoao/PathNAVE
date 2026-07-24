#!/bin/bash

python uncertainty_matching.py --provider openai --model gpt-4o \
  --matching_dir /work/Agent_benchmark/PathAgent/post_eval/TCGA_BRCA_subtype/Evidence/Matching \
  --atom_dir     /work/Agent_benchmark/PathAgent/post_eval/TCGA_BRCA_subtype/Evidence/Atom \
  --output_dir   /work/Agent_benchmark/PathAgent/post_eval/TCGA_BRCA_subtype/Evidence/Uncertainty_matching

python uncertainty_matching.py --provider openai --model gpt-4o \
  --matching_dir /work/Agent_benchmark/PathAgent/post_eval/TCGA_LUNG_Classification/Evidence/Matching \
  --atom_dir     /work/Agent_benchmark/PathAgent/post_eval/TCGA_LUNG_Classification/Evidence/Atom \
  --output_dir   /work/Agent_benchmark/PathAgent/post_eval/TCGA_LUNG_Classification/Evidence/Uncertainty_matching
