import os
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.distributions.categorical import Categorical

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

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Setup device and seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading environment '{args.env_id}' and model from '{args.model_path}'...")

    # 2. Setup Environment strictly for recording
    env = gym.make(args.env_id, render_mode="rgb_array")
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
    agent.load_state_dict(torch.load(args.model_path, map_location=device))
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
            logits = agent.actor(obs_tensor)
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