#!/bin/bash
#SBATCH --job-name=sac_mujoco_vlm
#SBATCH --output=/scratch/gilbreth/yang2739/logs/sac_mujoco_%j.out
#SBATCH --error=/scratch/gilbreth/yang2739/logs/sac_mujoco_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

export SCRATCH_DIR="/scratch/gilbreth/yang2739"
export PROJECT_DIR="$SCRATCH_DIR/VLM-RL-RUNS"
export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venv"
export PYTHONPATH="$PYTHONPATH:$HOME/CS490/VLM-RL/cleanrl"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl

module --force purge
cd "$HOME/CS490/VLM-RL"
uv sync

CMD="srun uv run python experiments/mujoco/sac_vlm.py \
  --env-id Reacher-v4 \
  --run-id $SLURM_JOB_ID \
  --seed 1 \
  --total-timesteps 1000000 \
  --vlm-goal 'A two-jointed robot arm minimizing the fingertip distance to a target red point.' \
  --vlm-n-frames 8 \
  --vlm-frame-every 2 \
  --vlm-clip-every 16 \
  --capture-video
"

if [ "$1" == "clip" ]; then
  CMD="$CMD --vlm-model-type clip --vlm-model-id openai/clip-vit-base-patch32 --video-every 500"
elif [ "$1" == "qwen" ]; then
  CMD="$CMD --vlm-model-type qwen --vlm-model-id Qwen/Qwen2-VL-2B-Instruct --video-every 100"
elif [ "$1" == "xclip" ]; then
  CMD="$CMD --vlm-model-type xclip --vlm-model-id microsoft/xclip-base-patch32  --video-every 100"
else
  echo "Invalid argument: $1. Use 'clip', 'qwen', or 'xclip'."
  exit 1
fi

echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Starting SAC in MuJoCo environment training..."
echo "Starting SAC + $1 in MuJoCo environment training..."
echo "Command: $CMD"
eval "$CMD"