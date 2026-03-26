#!/bin/bash
#SBATCH --job-name=sac_qwen_ant_training
#SBATCH --output=/scratch/gilbreth/yang2739/logs/sac_qwen_%j.out
#SBATCH --error=/scratch/gilbreth/yang2739/logs/sac_qwen_%j.err
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
mkdir -p "$PROJECT_DIR/hf_cache" "$PROJECT_DIR/uv_cache" "$PROJECT_DIR/logs"

export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl

module --force purge
export PATH="$HOME/.cargo/bin:$PATH"
unset VIRTUAL_ENV

echo "Running on node: $SLURM_NODELIST"
echo "Submitting cluster: gilbreth"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd "$HOME/Group-VLM-RL"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venv"
uv sync

CMD="srun uv run python experiments/ant/sac_qwen_multiframe.py \
  --env-id Ant-v4 \
  --total-timesteps 1000000 \
  --vlm-model-id Qwen/Qwen2-VL-2B-Instruct \
  --vlm-goal 'A four-legged ant robot walking forward to the right stably and upright on a flat surface.' \
  --vlm-n-frames 4 \
  --vlm-frame-every 4 \
  --vlm-clip-every 16 \
  --track \
  --capture-video \
  --wandb-project-name cleanrl-experiments"

echo "Starting SAC + Qwen in MuJoCo Ant environment training..."
echo "Command: $CMD"
eval "$CMD"