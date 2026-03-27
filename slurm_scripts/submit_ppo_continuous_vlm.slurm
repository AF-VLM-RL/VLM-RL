#!/bin/bash
#SBATCH --job-name=ppo_cont_vlm
#SBATCH --output=/scratch/gilbreth/sohn31/logs/ppo_continuous_vlm_8frames_clip_multi_goal_%j.out
#SBATCH --error=/scratch/gilbreth/sohn31/logs/ppo_continuous_vlm_8frames_clip_multi_goal_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

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

echo "Starting PPO continuous action + VLM (Walker2d-v4, X-CLIP video-text reward)..."
uv run python experiments/ppo_continuous_action_vlm.py \
  --seed 1 \
  --env-id Walker2d-v4 \
  --num-envs 8 \
  --num-steps 256 \
  --total-timesteps 1000000 \
  --vlm-backend xclip \
  --vlm-goals "a robot walking to the right with a smooth, steady gait" \
  --vlm-goal-weights 1.0 \
  --xclip-model-name "microsoft/xclip-base-patch32" \
  --vlm-max-frames 8 \
  --vlm-frame-stride 5 \
  --vlm-rollout-chunk-size 512 \
  --update-epochs 5 \
  --track \
  --wandb-project-name "cleanrl-experiments" \
  --capture-video
