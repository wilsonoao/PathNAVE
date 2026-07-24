#!/bin/bash
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

MAX_RETRIES=1

run_with_retry() {
    local name="$1"
    shift
    local attempt=0
    while [ $attempt -lt $MAX_RETRIES ]; do
        attempt=$((attempt + 1))
        echo "[${name}] attempt ${attempt}/${MAX_RETRIES} — $(date)"
        python run_qa_eval.py "$@"
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "[${name}] finished successfully."
            return 0
        fi
        echo "[${name}] exited with code ${exit_code}, restarting in 5s..."
        sleep 5
    done
    echo "[${name}] gave up after ${MAX_RETRIES} attempts."
    return 1
}

# Instance A: cases [0, 471)
run_with_retry "instance_a" \
    --json /work/Agent_dataset/TCGA/BRCA/subtype_qa_dataset.json \
    --slides /work/data/TCGA-BRCA-FS/CHIEF/picture_wsi \
    --output ./tcga_brca_subtype \
    --no_vision_supervisor \
    --openai-model o4-mini \
    --use-openai \
    --start-from 0 --end-at 471 


# # Instance B: cases [471, end)
# run_with_retry "instance_b" \
#     --json /work/Agent_dataset/TCGA/BRCA/subtype_qa_dataset.json \
#     --slides /work/data/TCGA-BRCA-FS/CHIEF/picture_wsi \
#     --output ./tcga_brca_subtype \
#     --no_vision_supervisor \
#     --openai-model o4-mini \
#     --use-openai \
#     --start-from 471 \
#     >> "$LOG_DIR/instance_b.log" 2>&1 &

wait
echo "Both instances finished. Logs: $LOG_DIR/instance_a.log  $LOG_DIR/instance_b.log"
