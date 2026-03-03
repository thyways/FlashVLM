#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1

# Evaluation benchmarks. ("videomme" "egoschema" "mvbench" "longvideobench_val_v" "mlvu_test")
TASKS=("videomme")

# Pretrained model path.
PRETRAINED="llava-hf/llava-onevision-qwen2-7b-ov-hf"

# Model arguments for llava_hf.
ATTN_IMPLEMENTATION=flash_attention_2
BASE_MODEL_ARGS="pretrained=$PRETRAINED,attn_implementation=$ATTN_IMPLEMENTATION"

MODEL_ARGS="$BASE_MODEL_ARGS"
for task in "${TASKS[@]}"; do
    echo "Evaluating task: $task"
    accelerate launch \
    --main_process_port 12346\
    --num_processes 8 \
    -m lmms_eval \
    --model llava_hf \
    --model_args "$MODEL_ARGS" \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "llava_ov" \
    --output_path ./logs/baseline 
done
