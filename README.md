# VLM–RL Experimentation Guide

This guide describes how to integrate a Vision–Language Model (VLM)–based reward signal into CleanRL single-file algorithm implementations. While PPO is used as the running example, the same integration pattern applies to most CleanRL scripts (e.g., DQN, SAC).

---

## 1. Environment setup

The repository already provides a `pyproject.toml` with the required core dependencies (PyTorch, Transformers, Gymnasium, etc.). Sync the environment and install CleanRL locally.

```bash
# Sync dependencies from pyproject.toml
uv sync

# Clone CleanRL and install it locally to access the reference implementations
git clone https://github.com/vwxyzjn/cleanrl.git
cd cleanrl
uv pip install .
cd ..

# (Optional) Login to Weights & Biases for experiment tracking
uv run wandb login
```

---

## 2. Prepare an experiment

CleanRL algorithms are implemented as standalone, single-file scripts. To modify an algorithm, copy the corresponding script into your local `experiments/` directory.

Example (PPO):

```bash
cp cleanrl/ppo.py experiments/vlm_ppo.py
```

You may instead copy other algorithms such as `dqn.py` or `sac.py` and apply the same integration steps described below.

---

## 3. VLM reward wrapper

You must define your own environment wrapper that uses a vision–language model (VLM) to compute the reward. This wrapper should live in the `src/` directory and follow the standard Gymnasium wrapper interface.

An example implementation is provided in:

```
src/wrappers.py
```

as the `VLMRewardWrapper` class. You are expected to use this file as a reference and adapt or extend it to support your own VLM or reward formulation.

In the provided example, `VLMRewardWrapper` uses a CLIP model (ViT-L/14) to compute a dense reward based on the visual similarity between the current rendered frame and a natural‑language goal description.

In short, the CleanRL algorithm should remain unchanged. All VLM‑based reward design and model selection must be implemented inside your custom wrapper in `src/wrappers.py`.

---

## 4. Integration into a CleanRL script

Your experiment file should apply **your own custom wrappers** (defined in `src/`) to the environment. The CleanRL training logic itself should remain unchanged. All VLM-based reward logic must be encapsulated inside your wrapper.

In this guide, `VLMRewardWrapper` is used as a concrete example.

In your experiment file (e.g., `experiments/vlm_ppo.py`), apply the following changes.

### 4.1 Import your wrapper(s)

Import the wrapper you implemented in `src/`. For example:

```python
from src.wrappers import VLMRewardWrapper
```

If you define multiple custom wrappers (e.g., different VLMs or reward formulations), select and apply them here.

### 4.2 Modify `make_env`

Wrap the base environment with your custom wrapper(s) before returning it. The environment must expose rendered RGB frames so that the wrapper can compute visual rewards.

```python
def make_env(env_id, idx, capture_video, run_name, vlm_goal, vlm_device):
    def thunk():
        # Enable RGB frame rendering for visual reward computation
        env = gym.make(env_id, render_mode="rgb_array")

        # ... standard CleanRL wrappers such as
        # RecordVideo and RecordEpisodeStatistics ...

        # Apply your custom VLM-based reward wrapper (example: VLMRewardWrapper)
        env = VLMRewardWrapper(
            env,
            goal=vlm_goal,
            device=vlm_device,
            skip_frames=16,
        )

        return env
    return thunk
```

Important:

* Your experiment file should only be responsible for *applying* wrappers.
* All VLM model loading, inference, and reward computation must be implemented inside the wrapper (e.g., `VLMRewardWrapper`).
* The wrapper must have access to the rendered frame returned by the environment.

---

## 5. Running the experiment

You can define the task purely through a natural-language goal without modifying the environment code. The goal string is passed to the wrapper and used by the VLM to compute rewards.

Example (PPO + CartPole):

```bash
uv run python experiments/vlm_ppo.py \
    --seed 1 \
    --env-id CartPole-v1 \
    --total-timesteps 500000 \
    --capture-video \
    --vlm-goal "a pole balancing upright on a cart"
```

Changing `--vlm-goal` allows you to redefine the task visually while keeping the same simulator and learning algorithm.

##
