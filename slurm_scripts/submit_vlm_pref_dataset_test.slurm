#!/bin/bash
#SBATCH --job-name=vlm_pref_dataset_test
#SBATCH --output=/scratch/gilbreth/sohn31/logs/vlm_pref_dataset_test_%j.out
#SBATCH --error=/scratch/gilbreth/sohn31/logs/vlm_pref_dataset_test_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

export SCRATCH_DIR="/scratch/gilbreth/sohn31"
export PROJECT_DIR="$SCRATCH_DIR/VLM-RL-RUNS"
mkdir -p "$PROJECT_DIR/hf_cache" "$PROJECT_DIR/uv_cache" "$PROJECT_DIR/logs"

export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export WANDB_MODE=offline

module --force purge
export PATH="$HOME/.cargo/bin:$PATH"
unset VIRTUAL_ENV

echo "Running on node: $SLURM_NODELIST"
echo "Submitting cluster: gilbreth"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd "$HOME/Group-VLM-RL"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venv"
uv sync

echo "Running VLM preference teacher once and saving debug dataset..."
srun uv run python experiments/vlm_preference_reward_walker2d.py \
  --reward-model-path pref_logs/reward_model.pt \
  --save-pref-dataset \
  --pref-dataset-path pref_logs/pref_dataset_debug.npz

