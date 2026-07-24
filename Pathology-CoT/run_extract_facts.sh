#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/extract_facts.py" \
       --provider         openai \
       --model            o4-mini \
       --reasoning_effort low \
       --input_dir  /work/Agent_benchmark/Pathology-CoT/result/TCGA_BRCA/test_total \
       --output_dir /work/Agent_benchmark/Pathology-CoT/post_eval/TCGA_BRCA
