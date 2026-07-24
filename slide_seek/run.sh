#!/bin/bash
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

MAX_RETRIES=30

cleanup() {
    echo ""
    echo "Interrupted — killing all instances..."
    trap - INT TERM
    kill 0
    exit 1
}
trap cleanup INT TERM

run_with_retry() {
    local name="$1"
    shift
    local attempt=0
    trap 'return 1' INT TERM
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

COMMON_ARGS=(
    --json /work/Agent_dataset/CAMELYON16/roi_qa_dataset.json
    --slides /work/Agent_dataset/CAMELYON16/slides
    --output /work/Agent_benchmark/slide_seek/CAMELYON16_detection
    --no_vision_supervisor
    --openai-model o4-mini
    --use-openai
)

# Instance A: cases [0, 314)
run_with_retry "instance_a" "${COMMON_ARGS[@]}" \
    --start-from 0 --end-at 128 \
    >> "$LOG_DIR/instance_a.log" 2>&1 &

# Instance B: cases [314, 628)
# run_with_retry "instance_b" "${COMMON_ARGS[@]}" \
#     --start-from 660 --end-at 1265 \
#     >> "$LOG_DIR/instance_b.log" 2>&1 &

# Instance C: cases [628, end)
# run_with_retry "instance_c" "${COMMON_ARGS[@]}" \
#     --start-from 800 \
#     >> "$LOG_DIR/instance_c.log" 2>&1 &

wait
echo "All instances finished. Logs: $LOG_DIR/"
