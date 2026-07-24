#!/bin/bash

python extract_facts.py --provider openai --model gpt-4o \
  --input_dir /work/Agent_benchmark/Pathology-CoT/result/CAMELYON16/test_total \
  --output_dir /work/Agent_benchmark/Pathology-CoT/post_eval/CAMELYON16_detection/Evidence/Atom \
  --reasoning_effort low 
  
# python extract_facts.py --provider openai --model gpt-4o \
#   --input_dir ../result/TCGA_BRCA/results \
#   --output_dir /work/Agent_benchmark/Pathology-CoT/post_eval/TCGA_BRCA_subtype/Evidence/Atom


python extract_facts.py --provider openai --model gpt-4o \
  --input_dir /work/Agent_benchmark/Pathology-CoT/result/TCGA_LUNG_LUSC_result/test/test_total \
  --output_dir /work/Agent_benchmark/Pathology-CoT/post_eval/TCGA_LUNG_Classification/Evidence/Atom \
  --reasoning_effort low 


python extract_facts.py --provider openai --model gpt-4o \
  --input_dir /work/Agent_benchmark/Pathology-CoT/result/TCGA_LUNG_result/test/test_total \
  --output_dir /work/Agent_benchmark/Pathology-CoT/post_eval/TCGA_LUNG_Classification/Evidence/Atom \
  --reasoning_effort low 
