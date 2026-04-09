#!/bin/bash
#SBATCH --job-name=sac_mujoco
#SBATCH --output=/scratch/gilbreth/yang2739/logs/sac_mujoco_%j.out
#SBATCH --error=/scratch/gilbreth/yang2739/logs/sac_mujoco_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

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

CMD="srun uv run python experiments/mujoco/sac.py \
  --env-id InvertedPendulum-v4 \
  --run-id $SLURM_JOB_ID \
  --seed 5 \
  --total-timesteps 1000000 \
  --track \
  --capture-video \
  --wandb-project-name cleanrl-experiments"

echo "Job ID: $SLURM_JOB_ID"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Starting SAC in MuJoCo environment training..."
echo "Command: $CMD"
eval "$CMD"