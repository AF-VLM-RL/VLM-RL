#!/bin/bash
#SBATCH --job-name=sac_ant_vlm
#SBATCH --output=/scratch/gilbreth/yang2739/logs/sac_ant_%j.out
#SBATCH --error=/scratch/gilbreth/yang2739/logs/sac_ant_%j.err
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
mkdir -p "$PROJECT_DIR/hf_cache"

export HF_HOME="$PROJECT_DIR/hf_cache"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl

module --force purge

echo "Job ID: $SLURM_JOB_ID"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd "$HOME/VLM-RL"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venv"
uv sync

CMD="srun uv run python experiments/mujoco/sac_vlm.py \
  --env-id Ant-v4 \
  --run-id $SLURM_JOB_ID \
  --seed 4 \
  --total-timesteps 1000000 \
  --vlm-goal 'A four-legged ant robot walking forward to the right, upright on a flat surface.' \
  --vlm-n-frames 4 \
  --vlm-frame-every 4 \
  --vlm-clip-every 16 \
  --track \
  --capture-video \
  --wandb-project-name cleanrl-experiments"

if [ "$1" == "clip" ]; then
  CMD="$CMD --vlm-model-type clip --vlm-model-id openai/clip-vit-base-patch32 --video-every 500"
elif [ "$1" == "qwen" ]; then
  CMD="$CMD --vlm-model-type qwen --vlm-model-id Qwen/Qwen2-VL-2B-Instruct --video-every 100"
else
  echo "Invalid argument: $1. Use 'clip' or 'qwen'."
  exit 1
fi

echo "Starting SAC + $1 in MuJoCo Ant environment training..."
echo "Command: $CMD"
eval "$CMD"