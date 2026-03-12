#!/bin/bash
#SBATCH --job-name=viz_warmstart_datasets
#SBATCH --output=/scratch/gilbreth/sohn31/logs/viz_warmstart_%j.out
#SBATCH --error=/scratch/gilbreth/sohn31/logs/viz_warmstart_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

export SCRATCH_DIR="/scratch/gilbreth/sohn31"
export PROJECT_DIR="$SCRATCH_DIR/VLM-RL-RUNS"
mkdir -p "$PROJECT_DIR/hf_cache" "$PROJECT_DIR/uv_cache" "$PROJECT_DIR/logs"

export HF_HOME="$PROJECT_DIR/hf_cache"
export UV_CACHE_DIR="$PROJECT_DIR/uv_cache"
export PYTHONNOUSERSITE=1

module --force purge
export PATH="$HOME/.cargo/bin:$PATH"
unset VIRTUAL_ENV

echo "Running on node: $SLURM_NODELIST"
echo "Submitting cluster: gilbreth"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd "$HOME/Group-VLM-RL"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/venv"
uv sync

# Override via sbatch --export=DQN_NPZ=...,SAC_NPZ=...,OUTPUT_DIR=...,MAX_IMAGES_PER_RATING=...
DQN_NPZ="${DQN_NPZ:-runs/warmstart_datasets/dqn_erl_vlm_cartpole.npz}"
SAC_NPZ="${SAC_NPZ:-runs/warmstart_datasets/sac_erl_vlm_cartpole.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/warmstart_dataset_viz}"
MAX_IMAGES_PER_RATING="${MAX_IMAGES_PER_RATING:-64}"

echo "DQN_NPZ: $DQN_NPZ"
echo "SAC_NPZ: $SAC_NPZ"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "MAX_IMAGES_PER_RATING: $MAX_IMAGES_PER_RATING"

echo "Starting warmstart dataset visualization..."
srun uv run python evaluations/visualize_warmstart_datasets.py \
  --dqn-npz "$DQN_NPZ" \
  --sac-npz "$SAC_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --max-images-per-rating "$MAX_IMAGES_PER_RATING"

