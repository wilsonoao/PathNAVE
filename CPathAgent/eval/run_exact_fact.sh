#!/bin/bash

# ── CPathAgent outputs ────────────────────────────────────────────────────────

python extract_facts_cpathagent.py --provider openai --model gpt-4o \
  --input_dir ../output/CAMELYON16 \
  --output_dir /work/Agent_benchmark/CPathAgent/post_eval/CAMELYON16_detection/Evidence/Atom

python extract_facts_cpathagent.py --provider openai --model gpt-4o \
  --input_dir /work/Agent_benchmark/CPathAgent/output/test_tcga_brca/TCGA_BRCA \
  --output_dir /work/Agent_benchmark/CPathAgent/post_eval/TCGA_BRCA_subtype/Evidence/Atom

python extract_facts_cpathagent.py --provider openai --model gpt-4o \
  --input_dir /work/Agent_benchmark/CPathAgent/output/test_tcga_luad/TCGA_LUAD \
  --output_dir /work/Agent_benchmark/CPathAgent/post_eval/TCGA_LUNG_Classification/Evidence/Atom


python extract_facts_cpathagent.py --provider openai --model gpt-4o \
  --input_dir /work/Agent_benchmark/CPathAgent/output/test_tcga_lusc/TCGA_LUSC \
  --output_dir /work/Agent_benchmark/CPathAgent/post_eval/TCGA_LUNG_Classification/Evidence/Atom
