#!/bin/bash

python extract_facts_pathagent.py --provider openai --model gpt-4o \
  --input_dir ../result/CAMELYON16/results \
  --desc_dir ../result/CAMELYON16/desc \
  --output_dir /work/Agent_benchmark/PathAgent/post_eval/CAMELYON16_detection/Evidence/Atom

python extract_facts_pathagent.py --provider openai --model gpt-4o \
  --input_dir ../result/TCGA_BRCA/results \
  --desc_dir ../result/TCGA_BRCA/desc \
  --output_dir /work/Agent_benchmark/PathAgent/post_eval/TCGA_BRCA_subtype/Evidence/Atom


python extract_facts_pathagent.py --provider openai --model gpt-4o \
  --input_dir ../result/TCGA_LUSC/results \
  --desc_dir ../result/TCGA_LUSC/desc \
  --output_dir /work/Agent_benchmark/PathAgent/post_eval/TCGA_LUNG_Classification/Evidence/Atom


python extract_facts_pathagent.py --provider openai --model gpt-4o \
  --input_dir ../result/TCGA_LUAD/results \
  --desc_dir ../result/TCGA_LUAD/desc \
  --output_dir /work/Agent_benchmark/PathAgent/post_eval/TCGA_LUNG_Classification/Evidence/Atom
