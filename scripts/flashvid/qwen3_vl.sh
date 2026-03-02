#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1

TASKS=("videomme")
PRETRAINED="Qwen/Qwen3-VL-8B-Instruct"

MAX_NUM_FRAMES=32
MIN_PIXELS=65536
MAX_PIXELS=262144
ATTN_IMPLEMENTATION=flash_attention_2

BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,attn_implementation=$ATTN_IMPLEMENTATION"
MODEL_ARGS="$BASE_MODEL_ARGS,interleave_visuals=False"

for task in "${TASKS[@]}"; do
    echo "Evaluating task: $task"
    COMPRESSOR=flashvid \
    FLASHVID_RETENTION_RATIO=0.25 \
    FLASHVID_DO_SEGMENT=true \
    FLASHVID_SEGMENT_THRESHOLD=0.9 \
    FLASHVID_MIN_SEGMENT_NUM=8 \
    FLASHVID_COMPLEMENTARY_SEGMENT=true \
    FLASHVID_TOKEN_SELECTION_METHOD=attn_div \
    FLASHVID_ALPHA=0.7 \
    FLASHVID_TEMPORAL_THRESHOLD=0.8 \
    FLASHVID_EXPANSION=1.25 \
    FLASHVID_PRUNING_LAYER=20 \
    FLASHVID_LLM_RETENTION_RATIO=0.3 \
    accelerate launch \
    --main_process_port 12346 \
    --num_processes 8 \
    -m lmms_eval \
    --model qwen3_vl \
    --model_args $MODEL_ARGS \
    --tasks $task \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "qwen3_vl_flashvid" \
    --output_path ./logs/flashvid
done
