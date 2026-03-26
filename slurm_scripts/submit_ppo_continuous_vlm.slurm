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

# echo "Starting PPO continuous action + VLM (Walker2d-v4, CLIP multi-goal weighted reward)..."
# srun uv run python experiments/ppo_continuous_action_vlm.py \
#   --seed 1 \
#   --env-id Walker2d-v4 \
#   --num-envs 8 \
#   --num-steps 256 \
#   --total-timesteps 1000000 \
#   --vlm-backend clip \
#   --vlm-goals "a robot standing tall and upright" "a robot taking a wide stride to the right" "a robot moving right with a sharply bent, raised knee" \
#   --vlm-goal-weights 0.1 0.1 0.8 \
#   --clip-model-name "openai/clip-vit-base-patch32" \
#   --vlm-layout grid \
#   --vlm-grid-size 8 \
#   --vlm-max-frames 8 \
#   --vlm-rollout-chunk-size 512 \
#   --update-epochs 5 \
#   --track \
#   --wandb-project-name "cleanrl-experiments" \
#   --capture-video


echo "Starting PPO continuous action + VLM (Walker2d-v4, Qwen video-input reward)..."
# NOTE: requires qwen-vl-utils in the environment for --vlm-input-format video.
srun uv run python experiments/ppo_continuous_action_vlm.py \
  --seed 1 \
  --env-id Walker2d-v4 \
  --num-envs 8 \
  --num-steps 256 \
  --total-timesteps 1000000 \
  --vlm-backend generate \
  --vlm-input-format video \
  --vlm-video-fps 8.0 \
  --vlm-goal "a 2D robot with a small central torso and two jointed legs but no arms, staying upright and walking steadily to the right over time without falling" \
  --vlm-model-name "Qwen/Qwen2-VL-72B-Instruct" \
  --vlm-layout grid \
  --vlm-grid-size 4 \
  --vlm-max-frames 16 \
  --vlm-frame-stride 5 \
  --vlm-rollout-chunk-size 32 \
  --update-epochs 5 \
  --track \
  --wandb-project-name "cleanrl-experiments" \
  --capture-video
