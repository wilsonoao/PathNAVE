#!/bin/bash

python extract_facts_slideseek.py --provider openai --model gpt-4o \
  --input_dir ../CAMELYON16_detection \
  --output_dir /work/Agent_benchmark/slide_seek/post_eval/CAMELYON16_detection/Evidence/Atom

python extract_facts_slideseek.py --provider openai --model gpt-4o \
  --input_dir ../tcga_brca_subtype \
  --output_dir /work/Agent_benchmark/slide_seek/post_eval/TCGA_BRCA_subtype/Evidence/Atom

python extract_facts_slideseek.py --provider openai --model gpt-4o \
  --input_dir ../tcga_luad_subtype \
  --output_dir /work/Agent_benchmark/slide_seek/post_eval/TCGA_LUNG_Classification/Evidence/Atom

python extract_facts_slideseek.py --provider openai --model gpt-4o \
  --input_dir ../tcga_lusc_subtype \
  --output_dir /work/Agent_benchmark/slide_seek/post_eval/TCGA_LUNG_Classification/Evidence/Atom
