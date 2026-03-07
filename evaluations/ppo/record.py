import os
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.distributions.categorical import Categorical


class RenderObservationWrapper(gym.Wrapper):
    """Return rgb_array frames as observations."""

    def __init__(self, env):
        super().__init__(env)
        obs, _ = self.env.reset()
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None; expected render_mode='rgb_array'.")
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=frame.shape, dtype=np.uint8)
        self._last_info = {}
        self._last_raw_obs = obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None on reset.")
        self._last_info = info
        self._last_raw_obs = obs
        return frame, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None on step.")
        self._last_info = info
        self._last_raw_obs = obs
        return frame, reward, terminated, truncated, info


@dataclass
class Args:
    model_path: str
    """the path to the saved best.pt file"""
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    seed: int = 1
    """seed of the evaluation"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    video_dir: str = "videos/eval"
    """the folder to save the recorded video"""

# 1. Redefine the PPO Agent exactly as it was during training
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def _flatten_obs(self, x):
        return x.view(x.shape[0], -1)

    def get_value(self, x):
        return self.critic(self._flatten_obs(x))

    def get_action_and_value(self, x, action=None):
        logits = self.actor(self._flatten_obs(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(self._flatten_obs(x))

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Setup device and seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading environment '{args.env_id}' and model from '{args.model_path}'...")

    # Inspect checkpoint to decide if training used rendered image observations.
    state_dict = torch.load(args.model_path, map_location=device)
    expected_obs_dim = state_dict["actor.0.weight"].shape[1]

    # 2. Setup Environment strictly for recording
    env = gym.make(args.env_id, render_mode="rgb_array")
    default_obs_dim = int(np.array(env.observation_space.shape).prod())
    if expected_obs_dim != default_obs_dim:
        env = RenderObservationWrapper(env)
        print(
            f"Detected rendered-observation checkpoint: expected_obs_dim={expected_obs_dim}, "
            f"default_obs_dim={default_obs_dim}. Using RenderObservationWrapper."
        )
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder=args.video_dir, 
        episode_trigger=lambda x: True # Record the episode
    )
    
    # CleanRL network expects VectorEnv attributes, so we map them here
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space

    # 3. Load Model
    agent = Agent(env).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    # 4. Play the game
    obs, info = env.reset(seed=args.seed)
    done = False
    total_reward = 0
    step = 0

    while not done:
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            # For PPO evaluation, we ask the actor for the action logits.
            # Exploitation only: take the action with the highest probability (argmax).
            logits = agent.actor(agent._flatten_obs(obs_tensor))
            action = torch.argmax(logits, dim=1).cpu().numpy()[0]
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

    env.close()
    
    print(f"--- Evaluation Complete ---")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward}")
    print(f"Video saved to: {args.video_dir}")