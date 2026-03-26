#!/bin/bash
#SBATCH --job-name=vlm_probe_walker2d
#SBATCH --output=/scratch/gilbreth/sohn31/logs/vlm_probe_walker2d_%j.out
#SBATCH --error=/scratch/gilbreth/sohn31/logs/vlm_probe_walker2d_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00

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

echo "Starting VLM reward probe for Walker2d-v4..."
srun uv run python scripts/test_vlm_reward_walker2d.py \
  --env-id Walker2d-v4 \
  --vlm-goal "a 2D robot with a small central torso and two jointed legs but no arms, staying upright and walking steadily to the right over time without falling" \
  --vlm-model-name "Qwen/Qwen2-VL-2B-Instruct" \
  --vlm-max-frames 16 \
  --vlm-layout grid \
  --vlm-grid-size 4 \
  --vlm-max-new-tokens 64 \
  --num-episodes 5 \
  --max-steps-per-episode 1000 \
  --out-dir "vlm_reward_probe"

