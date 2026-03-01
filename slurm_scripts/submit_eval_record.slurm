#!/bin/bash
#SBATCH --job-name=erl_vlm_eval_record
#SBATCH --output=/scratch/gilbreth/sohn31/logs/eval_record_%j.out
#SBATCH --error=/scratch/gilbreth/sohn31/logs/eval_record_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

# --- EVALUATION CONFIG ---
# Override these at submit-time if desired:
# sbatch --export=RUN_ID=CartPole-v1__ppo_erl_vlm__1__1772317659 submit_eval_record.slurm
RUN_ID="${RUN_ID:-CartPole-v1__ppo_erl_vlm__1__1772391649}"
MODEL_PATH="runs/${RUN_ID}/best_model.pt"
VIDEO_DIR="evaluations/ppo/videos/eval/${RUN_ID}"

echo "RUN_ID: ${RUN_ID}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "VIDEO_DIR: ${VIDEO_DIR}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: model checkpoint not found at $MODEL_PATH"
  exit 1
fi

echo "Starting evaluation recording..."
srun uv run python evaluations/ppo/record.py \
  --model-path "$MODEL_PATH" \
  --video-dir "$VIDEO_DIR" \
  --no-cuda

