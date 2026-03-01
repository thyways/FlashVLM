#!/bin/bash
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1

# Evaluation benchmarks.("videomme" "mvbench" "longvideobench_val_v" "egoschema")
TASKS=("videomme")

# Pretrained model path.
PRETRAINED="lmms-lab/LLaVA-Video-7B-Qwen2"

# Model arguments.
MAX_FRAMES_NUM=64
CONV_TEMPLATE=qwen_1_5
FORCE_SAMPLE=True
ADD_TIME_INSTRUCTION=False
MM_SPATIAL_POOL_MODE=average
MM_NEWLINE_POSITION=frame
ATTN_IMPLEMENTATION=flash_attention_2
BASE_MODEL_ARGS="pretrained=$PRETRAINED,conv_template=$CONV_TEMPLATE,mm_spatial_pool_mode=$MM_SPATIAL_POOL_MODE,mm_newline_position=$MM_NEWLINE_POSITION,max_frames_num=$MAX_FRAMES_NUM,attn_implementation=$ATTN_IMPLEMENTATION,force_sample=$FORCE_SAMPLE,add_time_instruction=$ADD_TIME_INSTRUCTION"

MODEL_ARGS="$BASE_MODEL_ARGS"
for task in "${TASKS[@]}"; do
    echo "Evaluating task: $task"
    accelerate launch \
        --main_process_port 12346 \
        --num_processes 8 \
        -m lmms_eval \
        --model llava_vid \
        --model_args $MODEL_ARGS \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "llava_vid" \
        --output_path ./logs/baseline
done
