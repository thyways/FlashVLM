#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HOME}/.cache/huggingface"

TASKS=("videomme")
PRETRAINED="Qwen/Qwen3-VL-8B-Instruct"
RETENTION_RATIOS=0.25
DO_SEGMENT=true
SEGMENT_THRESHOLD=0.9
MIN_SEGMENT_NUM=4
COMPLEMENTARY_SEGMENT=true
TOKEN_SELECTION_METHOD=attn_div
ALPHA=0.7
TEMPORAL_THRESHOLD=0.8
EXPANSION=1.25
PRUNING_LAYER=28
LLM_RETENTION_RATIO=0.1

MAX_NUM_FRAMES=32
ATTN_IMPLEMENTATION=flash_attention_2

BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION"
MODEL_ARGS="$BASE_MODEL_ARGS,interleave_visuals=False"

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running with retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        COMPRESSOR=flashvid \
        FLASHVID_RETENTION_RATIO=$retention_ratio \
        FLASHVID_DO_SEGMENT=$DO_SEGMENT \
        FLASHVID_SEGMENT_THRESHOLD=$SEGMENT_THRESHOLD \
        FLASHVID_MIN_SEGMENT_NUM=$MIN_SEGMENT_NUM \
        FLASHVID_COMPLEMENTARY_SEGMENT=$COMPLEMENTARY_SEGMENT \
        FLASHVID_TOKEN_SELECTION_METHOD=$TOKEN_SELECTION_METHOD \
        FLASHVID_ALPHA=$ALPHA \
        FLASHVID_TEMPORAL_THRESHOLD=$TEMPORAL_THRESHOLD \
        FLASHVID_EXPANSION=$EXPANSION \
        FLASHVID_PRUNING_LAYER=$PRUNING_LAYER \
        FLASHVID_LLM_RETENTION_RATIO=$LLM_RETENTION_RATIO \
        accelerate launch \
        --main_process_port 18888 \
        --num_processes 8 \
        -m lmms_eval \
        --model qwen3_vl \
        --model_args $MODEL_ARGS \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "qwen3_vl" \
        --output_path ./logs/flashvid
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
