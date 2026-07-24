#!/bin/bash

# ─── Define ranges to run in parallel (format: "start:end") ───────────────────
RANGES=(
    "0:60"
    "60:130"
)

# ─── Shared arguments ─────────────────────────────────────────────────────────
COMMON_ARGS=(
    --plip_lib_path /work/Agent_benchmark/PathAgent/plip
    --openai_model o4-mini
    --plip_ckpt vinid/plip
    --patho_r1_ckpt WenchuanZhang/Patho-R1-3B
    --descriptions_file /work/Agent_benchmark/PathAgent/result/CAMELYON16/desc/patches_descriptions.json
    --questions_file /work/Agent_dataset/CAMELYON16/subtype_qa_dataset.json
    --feature_dir /work/Agent_benchmark/PathAgent/result/CAMELYON16/img_features
    --patch_root /work/Agent_benchmark/PathAgent/result/CAMELYON16/patches_output
    --save_dir /work/Agent_benchmark/PathAgent/result/CAMELYON16/results
    --dataset_name "tcga_brca"
    --mask_first_retrieval
    --mask_root /work/Agent_benchmark/PathAgent/result/CAMELYON16/mask_tumor
    # --qwen_ckpt Qwen/Qwen3-4B
)

# ─── Log directory ────────────────────────────────────────────────────────────
LOG_DIR="/work/Agent_benchmark/PathAgent/result/CAMELYON16/logs"
mkdir -p "$LOG_DIR"

# ─── Ctrl-C handler: kill all child processes ─────────────────────────────────
PIDS=()
cleanup() {
    echo ""
    echo "[$(date '+%H:%M:%S')] Interrupted — killing all processes (PIDs: ${PIDS[*]})..."
    for PID in "${PIDS[@]}"; do
        kill "$PID" 2>/dev/null
    done
    wait 2>/dev/null
    echo "All processes stopped."
    exit 1
}
trap cleanup INT

# ─── Launch one process per range ─────────────────────────────────────────────

for RANGE in "${RANGES[@]}"; do
    START="${RANGE%%:*}"
    END="${RANGE##*:}"
    LOG_FILE="${LOG_DIR}/run_${START}_${END}.log"

    echo "[$(date '+%H:%M:%S')] Launching: start=${START}, end=${END} → ${LOG_FILE}"

    python pathagent.py \
        "${COMMON_ARGS[@]}" \
        --start_idx "$START" \
        --end_idx   "$END"   \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} processes launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."
echo ""

# ─── Wait and report exit codes ───────────────────────────────────────────────
ALL_OK=true
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    RANGE="${RANGES[$i]}"
    wait "$PID"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ range ${RANGE} (PID ${PID}) finished successfully."
    else
        echo "[$(date '+%H:%M:%S')] ✗ range ${RANGE} (PID ${PID}) exited with code ${EXIT_CODE}."
        ALL_OK=false
    fi
done

echo ""
$ALL_OK && echo "All ranges completed successfully." || echo "Some ranges failed — check logs in ${LOG_DIR}."
