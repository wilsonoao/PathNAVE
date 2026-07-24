#!/bin/bash
echo "Launching 1 parallel tasks..."

PROJECT_ROOT="/work/Agent_benchmark/PathAgent/quilt-llava" # change this to your project root directory
DATASET_ROOT="/work/Agent_benchmark/PathAgent/result/CAMELYON16" # change this to your dataset root directory
# PROJECT_ROOT="/work/Agent_benchmark/PathAgent/quilt-llava" # change this to your project root directory
# DATASET_ROOT="/work/Agent_benchmark/PathAgent/result/TCGA_LUAD" # change this to your dataset root directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p $LOG_DIR

GPUS=(0 0) # change this to your GPU IDs for each task


for i in $(seq 1 2)
do
    GPU_ID=${GPUS[$((i-1))]}
    echo "Starting Task $i on GPU $GPU_ID..."

    CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_ROOT/description_generation.py" \
        --model-path "wisdomik/Quilt-Llava-v1.5-7b" \
        --image-dir "$DATASET_ROOT/patches_output" \
        --output-json "$DATASET_ROOT/desc/patches_descriptions${i}.json" \
        --slide-list "$DATASET_ROOT/split_name/slides_part${i}.txt" \
        --load-4bit > "$LOG_DIR/task${i}.log" 2>&1 &
done

echo "✅ All tasks started successfully. Please check $LOG_DIR for logs."



# #!/bin/bash
# echo "Launching parallel tasks on single GPU..."

# PROJECT_ROOT="/work/Agent_benchmark/PathAgent/quilt-llava"
# DATASET_ROOT="/work/data/TCGA-BRCA-FS/CHIEF/crop_wsi"
# LOG_DIR="$PROJECT_ROOT/logs"
# mkdir -p $LOG_DIR

# # ✅ 單 GPU
# GPU_ID=0

# # ✅ 開 3 個 process（你目前 VRAM 很夠）
# NUM_TASKS=3

# # ✅ 避免 CPU thread 爆掉
# export OMP_NUM_THREADS=4

# for i in $(seq 1 $NUM_TASKS)
# do
#     echo "Starting Task $i on GPU $GPU_ID..."

#     CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_ROOT/description_generation.py" \
#         --model-path "wisdomik/Quilt-Llava-v1.5-7b" \
#         --image-dir "$DATASET_ROOT/patches_output" \
#         --output-json "$DATASET_ROOT/desc/patches_descriptions${i}.json" \
#         --slide-list "$DATASET_ROOT/split_name/slides_part${i}.txt" \
#         --load-4bit > "$LOG_DIR/task${i}.log" 2>&1 &
# done

# # ✅ 等全部跑完（很重要）
# wait

# echo "✅ All tasks finished."
