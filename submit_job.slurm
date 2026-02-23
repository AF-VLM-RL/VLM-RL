#!/bin/bash
#SBATCH --job-name=vlm_circle
#SBATCH --output=logs/a100_%j.out
#SBATCH --error=logs/a100_%j.err
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH -q standby
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# --- SETUP ---
module purge
# Note: Ensure 'uv' is in your PATH. If it's a module on Gilbreth, load it here:
# module load uv 

# Identify the environment
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# --- EXECUTION ---
# 'uv sync' ensures the virtual environment matches your requirements exactly.
# 'uv run' will then use that environment automatically.

echo "Syncing dependencies with uv..."
uv sync

# Define the root of your project
PROJECT_ROOT=$SLURM_SUBMIT_DIR

echo "Starting training from $PROJECT_ROOT..."

# srun uv run python "$PROJECT_ROOT/cleanrl/cleanrl/ppo.py" \
#     --seed 3 \
#     --env-id CartPole-v1 \
#     --total-timesteps 50000 \

srun uv run python experiments/ppo_vlm.py \
    --seed 1 \
    --env-id CartPole-v1 \
    --total-timesteps 50000 \
    --vlm-goal "a pole balancing upright on a cart"