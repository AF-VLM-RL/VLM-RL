#!/bin/bash
#SBATCH --job-name=vlm_pref_walker2d
#SBATCH --output=/scratch/gilbreth/sohn31/logs/vlm_pref_walker2d_%j.out
#SBATCH --error=/scratch/gilbreth/sohn31/logs/vlm_pref_walker2d_%j.err
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

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

echo "Starting VLM preference loop (teacher + student PPO)..."

# Number of teacher–student iterations
NUM_ITERS=1

# Paths shared between teacher and student
REWARD_MODEL_PATH="pref_logs/reward_model.pt"
POLICY_PATH="runs/ppo_continuous_action_reward_model_latest.cleanrl_model"

for ((it=1; it<=NUM_ITERS; it++)); do
  echo "=== Iteration ${it}/${NUM_ITERS} ==="

  # 1) Teacher: train/update reward model with VLM preferences.
  if [[ "$it" -eq 1 ]]; then
    echo "Teacher: collecting trajectories with random policy."
    srun uv run python experiments/vlm_preference_reward_walker2d.py \
      --reward-model-path "$REWARD_MODEL_PATH"
  else
    echo "Teacher: collecting trajectories with learned policy at $POLICY_PATH."
    srun uv run python experiments/vlm_preference_reward_walker2d.py \
      --reward-model-path "$REWARD_MODEL_PATH" \
      --policy-path "$POLICY_PATH"
  fi

  # 2) Student: PPO with learned reward model, saving latest policy.
  echo "Student: training PPO with reward model $REWARD_MODEL_PATH."
  srun uv run python experiments/ppo_continuous_action_reward_model.py \
    --reward-model-path "$REWARD_MODEL_PATH" \
    --save-model True

done

