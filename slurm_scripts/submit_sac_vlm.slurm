#!/bin/bash
#SBATCH --job-name=sac_vlm_training
#SBATCH --output=/scratch/gilbreth/sohn31/logs/sac_vlm_%j.out
#SBATCH --error=/scratch/gilbreth/sohn31/logs/sac_vlm_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

export SCRATCH_DIR="/scratch/gilbreth/sohn31"
export PROJECT_DIR="$SCRATCH_DIR/VLM-RL-RUNS"
mkdir -p "$PROJECT_DIR/hf_cache" "$PROJECT_DIR/uv_cache" "$PROJECT_DIR/logs"

export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export PYTHONNOUSERSITE=1

module --force purge
export PATH="$HOME/.cargo/bin:$PATH"
unset VIRTUAL_ENV

echo "Running on node: $SLURM_NODELIST"
echo "Submitting cluster: gilbreth"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd "$HOME/Group-VLM-RL"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venv"
uv sync

echo "Starting SAC + VLM training..."
srun uv run python experiments/sac_vlm.py \
  --seed 1 \
  --env-id CartPole-v1 \
  --total-timesteps 50000 \
  --vlm-goal "a pole balancing upright on a cart"

