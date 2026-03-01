export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1

# Evaluation benchmarks.("videomme" "egoschema" "mvbench" "longvideobench_val_v" "mlvu_test")
TASKS=("videomme")

# Pretrained model path.
PRETRAINED="Qwen/Qwen3-VL-8B-Instruct"

# Model arguments.
MAX_NUM_FRAMES=16
# Configurable pixel constraints.
MIN_PIXELS=65536 # 64*32*32
MAX_PIXELS=262144 # 256*32*32
ATTN_IMPLEMENTATION=flash_attention_2
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,max_pixels=$MAX_PIXELS,attn_implementation=$ATTN_IMPLEMENTATION"

MODEL_ARGS="$BASE_MODEL_ARGS,interleave_visuals=False"
for task in "${TASKS[@]}"; do
    echo "Evaluating task: $task"
    accelerate launch \
    --main_process_port 12346 \
    --num_processes 8 \
    -m lmms_eval \
    --model qwen3_vl \
    --model_args $MODEL_ARGS \
    --tasks $task \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "qwen3_vl" \
    --output_path ./logs/baseline
done