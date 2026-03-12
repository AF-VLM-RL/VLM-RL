#!/bin/bash
#SBATCH --job-name=dqn_drawing
#SBATCH --output=/scratch/gilbreth/sohn31/logs/dqn_drawing_%j.out
#SBATCH --error=/scratch/gilbreth/sohn31/logs/dqn_drawing_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

export SCRATCH_DIR="/scratch/gilbreth/sohn31"
export PROJECT_DIR="$SCRATCH_DIR/VLM-RL-RUNS"
export VLM_RL_RUN_DIR="$PROJECT_DIR/runs"
mkdir -p "$PROJECT_DIR/hf_cache" "$PROJECT_DIR/uv_cache" "$PROJECT_DIR/logs" "$VLM_RL_RUN_DIR"

export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export PYTHONNOUSERSITE=1

module --force purge
export PATH="$HOME/.cargo/bin:$PATH"
unset VIRTUAL_ENV

echo "Running on node: $SLURM_NODELIST"
cd "$HOME/Group-VLM-RL"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venv"
uv sync

echo "Starting DQN drawing (CLIP-guided circle) training..."
srun uv run python experiments/dqn_drawing.py \
  --seed 1 \
  --total-timesteps 500000 \
  --save-eval-freq 10000
