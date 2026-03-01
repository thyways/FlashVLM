#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1

# Evaluation benchmarks.("videomme" "egoschema" "mvbench" "longvideobench_val_v" "mlvu_test")
TASKS=("videomme")

# Pretrained model path.
PRETRAINED="Qwen/Qwen2.5-VL-7B-Instruct"

# Model arguments.
MAX_NUM_FRAMES=16
# Configurable pixel constraints.
MIN_PIXELS=50176 # 64*28*28
MAX_PIXELS=200704 # 256*28*28
ATTN_IMPLEMENTATION=flash_attention_2
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,attn_implementation=$ATTN_IMPLEMENTATION"

MODEL_ARGS="$BASE_MODEL_ARGS,interleave_visuals=False"
for task in "${TASKS[@]}"; do
    echo "Evaluating task: $task"
    accelerate launch \
    --main_process_port 12346 \
    --num_processes 1 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args $MODEL_ARGS \
    --tasks $task \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "qwen2_5_vl" \
    --output_path ./logs/baseline
done