import os
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro

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

# 1. Redefine the Network exactly as it was during training
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 120)),
            nn.LayerNorm(120),
            nn.ReLU(),
            layer_init(nn.Linear(120, 84)),
            nn.LayerNorm(84),
            nn.ReLU(),
            layer_init(nn.Linear(84, env.single_action_space.n)),
        )

    def forward(self, x):
        return self.network(x)

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
    q_network = QNetwork(env).to(device)
    q_network.load_state_dict(torch.load(args.model_path, map_location=device))
    q_network.eval()

    # 4. Play the game
    obs, info = env.reset(seed=args.seed)
    done = False
    total_reward = 0
    step = 0

    while not done:
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_network(obs_tensor)
            # Exploitation only: always pick the best action
            action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

    env.close()
    
    print(f"--- Evaluation Complete ---")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward}")
    print(f"Video saved to: {args.video_dir}")