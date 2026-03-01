#!/bin/bash
#SBATCH --job-name=erl_vlm_dqn_training
#SBATCH --output=/scratch/gilbreth/sohn31/logs/vlm_%j.out
#SBATCH --error=/scratch/gilbreth/sohn31/logs/vlm_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1           # Set to gpu:4 for multi-GPU jobs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G             # Reduced to improve scheduling speed
#SBATCH --time=04:00:00

# --- GILBRETH SCRATCH CONFIG ---
# Keep heavy caches/env data on /scratch/gilbreth
export SCRATCH_DIR="/scratch/gilbreth/sohn31"
export PROJECT_DIR="$SCRATCH_DIR/VLM-RL-RUNS"
mkdir -p "$PROJECT_DIR/hf_cache" "$PROJECT_DIR/uv_cache" "$PROJECT_DIR/logs"

# Redirect Hugging Face and UV to scratch
export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export PYTHONNOUSERSITE=1 # Prevent loading packages from ~/.local/

# --- SETUP ---
module --force purge
# Ensure 'uv' is in your path (adjust if you installed it elsewhere)
export PATH="$HOME/.cargo/bin:$PATH"
unset VIRTUAL_ENV

# Identify the environment
echo "Running on node: $SLURM_NODELIST"
echo "Submitting cluster: gilbreth"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# --- EXECUTION ---
echo "Syncing dependencies in scratch..."
# Run from this repository directory (where pyproject.toml lives)
cd "$HOME/Group-VLM-RL"

# Use uv to sync, but force the environment to be created on scratch
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venv"
uv sync

echo "Starting VLM training..."

# # PPO with ERL
# srun uv run python experiments/ppo_erl_vlm.py \
#     --seed 1 \
#     --env-id CartPole-v1 \
#     --total-timesteps 50000 \
#     --vlm-goal "a pole balancing upright on a cart"


# DQN with ERL
srun uv run python experiments/dqn_erl_vlm.py \
    --seed 1 \
    --env-id CartPole-v1 \
    --total-timesteps 50000 \
    --vlm-goal "a pole balancing upright on a cart"

# # DQN with VLM
# srun uv run python experiments/dqn_vlm.py \
#     --seed 1 \
#     --env-id CartPole-v1 \
#     --total-timesteps 50000 \
#     --vlm-goal "a pole balancing upright on a cart"

# # PQN with VLM
# srun uv run python experiments/pqn_vlm.py \
#     --seed 1 \
#     --env-id CartPole-v1 \
#     --total-timesteps 50000 \
#     --vlm-goal "a pole balancing upright on a cart"