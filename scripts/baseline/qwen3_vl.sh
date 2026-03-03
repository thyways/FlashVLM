#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1

# Evaluation benchmarks.("videomme" "egoschema" "mvbench" "longvideobench_val_v" "mlvu_test")
TASKS=("videomme")

# Pretrained model path.
PRETRAINED="Qwen/Qwen3-VL-8B-Instruct"

# Model arguments.
MAX_NUM_FRAMES=32
ATTN_IMPLEMENTATION=flash_attention_2
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION"
USE_VLLM_CHAT_FOR_METRICS="${USE_VLLM_CHAT_FOR_METRICS:-0}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

for task in "${TASKS[@]}"; do
    echo "Evaluating task: $task"
    if [ "$USE_VLLM_CHAT_FOR_METRICS" = "1" ]; then
        # vllm_chat reports TTFT/TPOT in logs and throughput summary (avg_ttft/avg_tpot).
        MODEL_ARGS="model=$PRETRAINED,tensor_parallel_size=$VLLM_TP_SIZE,gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION,max_frame_num=$MAX_NUM_FRAMES,disable_log_stats=False,trust_remote_code=True"
        python -m lmms_eval \
        --model vllm_chat \
        --model_args "$MODEL_ARGS" \
        --tasks "$task" \
        --batch_size 1 \
        --verbosity INFO \
        --log_samples \
        --log_samples_suffix "qwen3_vl_vllm_chat" \
        --output_path ./logs/baseline
    else
        MODEL_ARGS="$BASE_MODEL_ARGS,interleave_visuals=False"
        accelerate launch \
        --main_process_port 12346 \
        --num_processes 8 \
        -m lmms_eval \
        --model qwen3_vl \
        --model_args "$MODEL_ARGS" \
        --tasks "$task" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "qwen3_vl" \
        --output_path ./logs/baseline
    fi
done
