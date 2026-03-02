#!/bin/bash
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
    COMPRESSOR=visionzip \
    VISIONZIP_R_RATIO=0.2 \
    VISIONZIP_DOMINANT_RATIO=0.6 \
    VISIONZIP_K_NEIGHBORS=5 \
    accelerate launch \
    --main_process_port 12346 \
    --num_processes 8 \
    -m lmms_eval \
    --model qwen3_vl \
    --model_args $MODEL_ARGS \
    --tasks $task \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "qwen3_vl_visionzip" \
    --output_path ./logs/visionzip
done
