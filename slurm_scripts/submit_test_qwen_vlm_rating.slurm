#!/bin/bash
#SBATCH --job-name=qwen_vlm_rating_test
#SBATCH --output=/scratch/gilbreth/sohn31/logs/qwen_rating_%j.out
#SBATCH --error=/scratch/gilbreth/sohn31/logs/qwen_rating_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

# --- GILBRETH SCRATCH CONFIG ---
export SCRATCH_DIR="/scratch/gilbreth/sohn31"
export PROJECT_DIR="$SCRATCH_DIR/VLM-RL-RUNS"
mkdir -p "$PROJECT_DIR/hf_cache" "$PROJECT_DIR/uv_cache" "$PROJECT_DIR/logs"

export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export PYTHONNOUSERSITE=1

# --- SETUP ---
module --force purge
export PATH="$HOME/.cargo/bin:$PATH"
unset VIRTUAL_ENV

echo "Running on node: $SLURM_NODELIST"
echo "Submitting cluster: gilbreth"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd "$HOME/Group-VLM-RL"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venv"
uv sync

# --- TEST CONFIG (override via --export=...) ---
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
GOAL="${GOAL:-a pole balancing upright on a cart}"
ENV_ID="${ENV_ID:-CartPole-v1}"
SEED="${SEED:-1}"
RANDOM_STEPS="${RANDOM_STEPS:-20}"
CAPTURE_COUNT="${CAPTURE_COUNT:-4}"
MAX_IMAGES="${MAX_IMAGES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/vlm_rating_debug}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DEVICE="${DEVICE:-cuda}" # cuda|cpu|auto
IMAGE_PATH="${IMAGE_PATH:-}" # Optional local image path
IMAGE_DIR="${IMAGE_DIR:-}" # Optional directory of image paths
VLM_DO_SAMPLE="${VLM_DO_SAMPLE:-true}"
VLM_TEMPERATURE="${VLM_TEMPERATURE:-0.2}"
VLM_TOP_P="${VLM_TOP_P:-0.9}"
VLM_COLLAPSE_THRESHOLD="${VLM_COLLAPSE_THRESHOLD:-0.8}"

echo "MODEL_NAME: $MODEL_NAME"
echo "GOAL: $GOAL"
echo "ENV_ID: $ENV_ID"
echo "SEED: $SEED"
echo "RANDOM_STEPS: $RANDOM_STEPS"
echo "CAPTURE_COUNT: $CAPTURE_COUNT"
echo "MAX_IMAGES: $MAX_IMAGES"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "DEVICE: $DEVICE"
echo "VLM_DO_SAMPLE: $VLM_DO_SAMPLE"
echo "VLM_TEMPERATURE: $VLM_TEMPERATURE"
echo "VLM_TOP_P: $VLM_TOP_P"
echo "VLM_COLLAPSE_THRESHOLD: $VLM_COLLAPSE_THRESHOLD"

CMD="uv run python evaluations/test_qwen_vlm_rating.py \
  --model-name \"$MODEL_NAME\" \
  --goal \"$GOAL\" \
  --env-id \"$ENV_ID\" \
  --seed \"$SEED\" \
  --random-steps \"$RANDOM_STEPS\" \
  --capture-count \"$CAPTURE_COUNT\" \
  --max-images \"$MAX_IMAGES\" \
  --output-dir \"$OUTPUT_DIR\" \
  --max-new-tokens \"$MAX_NEW_TOKENS\" \
  --vlm-temperature \"$VLM_TEMPERATURE\" \
  --vlm-top-p \"$VLM_TOP_P\" \
  --vlm-collapse-threshold \"$VLM_COLLAPSE_THRESHOLD\" \
  --device \"$DEVICE\""

if [ -n "$IMAGE_PATH" ]; then
  CMD="$CMD --image-path \"$IMAGE_PATH\""
fi

if [ -n "$IMAGE_DIR" ]; then
  CMD="$CMD --image-dir \"$IMAGE_DIR\""
fi

if [ "${VLM_DO_SAMPLE,,}" = "true" ] || [ "$VLM_DO_SAMPLE" = "1" ]; then
  CMD="$CMD --vlm-do-sample"
else
  CMD="$CMD --no-vlm-do-sample"
fi

echo "Starting Qwen VLM rating test..."
srun bash -lc "$CMD"

