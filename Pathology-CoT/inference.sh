#!/bin/bash

trap 'echo "Interrupted, killing all workers..."; kill 0; exit 1' INT TERM

BASE_DATA_DIR=/work/Agent_benchmark/Pathology-CoT/result/CAMELYON16
BASE_OUT_DIR=/work/Agent_benchmark/Pathology-CoT/result/CAMELYON16/test
MODEL=WenchuanZhang/Patho-R1-3B
MODEL_SUMMARY=o4-mini
QA_FILE=/work/Agent_dataset/CAMELYON16/roi_qa_dataset.json

LOG1="worker1.log"
LOG2="worker2.log"

python pathology-o3/think_ln_classify.py \
    -d "$BASE_DATA_DIR" \
    -o "$BASE_OUT_DIR" \
    -m "$MODEL" \
    -m_s "$MODEL_SUMMARY" \
    -n 1 \
    --qa_file "$QA_FILE" \
    --start 0 --end 130 > "$LOG1" 2>&1 &
PID1=$!
echo "[Worker 1] PID=$PID1  cases 0-450 -> $LOG1"

# python pathology-o3/think_ln_classify.py \
#     -d "$BASE_DATA_DIR" \
#     -o "$BASE_OUT_DIR" \
#     -m "$MODEL" \
#     -m_s "$MODEL_SUMMARY" \
#     -n 1 \
#     --qa_file "$QA_FILE" \
#     --start 65 > "$LOG2" 2>&1 &
# PID2=$!
# echo "[Worker 2] PID=$PID2  cases 450+  -> $LOG2"

wait $PID1
STATUS1=$?
# wait $PID2
# STATUS2=$?

echo "[Worker 1] exit status: $STATUS1"
# echo "[Worker 2] exit status: $STATUS2"

if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
    echo "One or more workers failed."
    exit 1
fi
echo "All workers finished successfully."
