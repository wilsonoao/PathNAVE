#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
# ── Mixed backend: OpenAI for text, HF for vision ────────────────────────────
# Run all 3 segments in parallel (background processes)

# Kill all background child processes on Ctrl-C
trap 'echo "[run_main.sh] Interrupted – killing all segments …"; kill 0; exit 1' INT TERM

COMMON=(
    --dataset /work/Agent_dataset/TCGA/LUNG_LUAD/subtype_qa_dataset.json
    --slide-dir /work/data/TCGA-LUAD-FS/CHIEF/picture_wsi
    --qa-set TCGA
    --tissue luad
    --text-backend openai
    --text-openai o4-mini
    --vlm-backend hf
    --vlm-hf WenchuanZhang/Patho-R1-3B
    --hf-device cuda
    --output-dir /work/Agent_benchmark/CPathAgent/output/test_tcga_luad
    # --hf-compile 
    --hf-flash-attn
)

echo "[run_main.sh] Starting segment 1: 0-330"
python evaluate_camelyon16.py "${COMMON[@]}" --start 800   --end 1000   > logs/seg1.log 2>&1 &

echo "[run_main.sh] Starting segment 2: 330-660"
python evaluate_camelyon16.py "${COMMON[@]}" --start 1000 --end 1280  > logs/seg2.log 2>&1 &

# echo "[run_main.sh] Starting segment 3: 660-941"
# python evaluate_camelyon16.py "${COMMON[@]}" --start 800 --end 1261   > logs/seg3.log 2>&1 &

echo "[run_main.sh] All 2 segments launched. Waiting …"
wait
echo "[run_main.sh] All segments finished."
