#!/bin/bash
#SBATCH --job-name=sac_mujoco_vlm
#SBATCH --output=/scratch/gilbreth/yang2739/logs/sac_mujoco_%j.out
#SBATCH --error=/scratch/gilbreth/yang2739/logs/sac_mujoco_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

export SCRATCH_DIR="/scratch/gilbreth/yang2739"
export PROJECT_DIR="$SCRATCH_DIR/VLM-RL-RUNS"
export HF_HUB_OFFLINE=1
export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venv"
export PYTHONPATH="$PYTHONPATH:$HOME/CS490/VLM-RL/cleanrl"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl

module --force purge
cd "$HOME/CS490/VLM-RL"
uv sync

CMD="srun uv run python experiments/mujoco/sac_vlm.py \
  --env-id Ant-v4 \
  --run-id $SLURM_JOB_ID \
  --seed 1 \
  --total-timesteps 1000000 \
  --vlm-goal 'A four-legged ant robot walking forward to the right, upright on a flat surface.' \
  --vlm-n-frames 8 \
  --vlm-frame-every 2 \
  --vlm-clip-every 16 \
  --capture-video \
"

if [ "$1" == "clip" ]; then
  CMD="$CMD --vlm-model-type clip --vlm-model-id openai/clip-vit-base-patch32 --video-every 500"
elif [ "$1" == "xclip" ]; then
  CMD="$CMD --vlm-model-type xclip --vlm-model-id microsoft/xclip-base-patch32  --video-every 100"
elif [ "$1" == "qwen2" ]; then
  CMD="$CMD --vlm-model-type qwen2 --vlm-model-id Qwen/Qwen2-VL-7B-Instruct --video-every 100"
elif [ "$1" == "qwen3" ]; then
  CMD="$CMD --vlm-model-type qwen3 --vlm-model-id Qwen/Qwen3-VL-8B-Instruct --video-every 100"
else
  echo "Invalid argument: $1. Use 'clip', 'qwen2', 'qwen3', or 'xclip'."
  exit 1
fi

echo "Command: $CMD"
echo "Starting batch script:environment=mujoco, algorithm=sac, model=$1, \
  device=$(nvidia-smi --query-gpu=name --format=csv,noheader), job_id=$SLURM_JOB_ID"
eval "$CMD"