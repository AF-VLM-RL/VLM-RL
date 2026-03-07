"""
CleanRL-style DQN for the hand-drawing circle environment.

Uses a CNN Q-network for 2-channel (canvas + cursor) image observations and
trains with CLIP-based rewards (judging how circle-like the drawing is).
"""

import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CLEANRL_ROOT = os.path.join(_PROJECT_ROOT, "cleanrl")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _CLEANRL_ROOT not in sys.path:
    sys.path.insert(0, _CLEANRL_ROOT)

from cleanrl_utils.buffers import ReplayBuffer  # pyright: ignore[reportMissingImports]
from src.drawing_env import DrawingEnv


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    save_eval_freq: int = 10000
    """Save evaluation drawing every N steps (0 disables)"""
    log_freq: int = 1000
    """Print progress every N steps (0 disables periodic logging)"""
    eval_save_path: str = "runs/dqn_drawing_eval"
    """Directory to save evaluation drawings (relative to project root)"""
    run_dir_base: Optional[str] = None
    """Base directory for runs."""

    total_timesteps: int = 500_000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 5000
    train_frequency: int = 10


def make_env(seed: int):
    def thunk():
        env = DrawingEnv()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def _get_run_dir_base(run_dir_base: Optional[str]) -> str:
    if run_dir_base is not None:
        return run_dir_base
    if base := os.environ.get("VLM_RL_RUN_DIR"):
        return base
    if project_dir := os.environ.get("PROJECT_DIR"):
        return os.path.join(project_dir, "runs")
    return os.path.join(_PROJECT_ROOT, "runs")


class CnnQNetwork(nn.Module):
    """CNN Q-network for (C, H, W) image observations."""

    def __init__(self, env):
        super().__init__()
        c, h, w = env.single_observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = int(np.prod(self.cnn(dummy).shape[1:]))
        self.head = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, env.single_action_space.n),
        )

    def forward(self, x):
        x = x.float() / 255.0
        feat = self.cnn(x)
        return self.head(feat)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs not supported (DrawingEnv loads CLIP per instance)"

    run_name = f"DrawingCircle__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir_base = _get_run_dir_base(args.run_dir_base)
    run_dir = os.path.join(run_dir_base, run_name)
    os.makedirs(run_dir, exist_ok=True)

    eval_save_path = (
        args.eval_save_path
        if os.path.isabs(args.eval_save_path)
        else os.path.join(_PROJECT_ROOT, args.eval_save_path)
    )

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
        )
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args.seed + i) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    q_network = CnnQNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = CnnQNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    eval_env = DrawingEnv() if args.save_eval_freq > 0 else None
    if eval_env and args.save_eval_freq > 0:
        os.makedirs(eval_save_path, exist_ok=True)

    recent_returns = []
    max_recent = 20

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    print(f"Starting DQN training: {args.total_timesteps} steps, log every {args.log_freq}")

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            q_values = q_network(torch.tensor(obs, device=device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ep_ret = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    recent_returns.append(ep_ret)
                    if len(recent_returns) > max_recent:
                        recent_returns.pop(0)
                    writer.add_scalar("charts/episodic_return", ep_ret, global_step)
                    writer.add_scalar("charts/episodic_length", ep_len, global_step)
        elif "episode" in infos:
            for i, done in enumerate(infos.get("_episode", [])):
                if done:
                    ep_ret = infos["episode"]["r"][i].item()
                    ep_len = infos["episode"]["l"][i].item()
                    recent_returns.append(ep_ret)
                    if len(recent_returns) > max_recent:
                        recent_returns.pop(0)
                    writer.add_scalar("charts/episodic_return", ep_ret, global_step)
                    writer.add_scalar("charts/episodic_length", ep_len, global_step)

        real_next_obs = next_obs.copy()
        final_observations = infos.get("final_observation")
        final_observation_mask = infos.get("_final_observation")
        for idx, trunc in enumerate(truncations):
            if not trunc:
                continue
            if final_observations is None:
                continue
            if final_observation_mask is None or final_observation_mask[idx]:
                real_next_obs[idx] = final_observations[idx]
        # ReplayBuffer expects list of per-env dicts; VectorEnv returns batch dict
        final_infos = infos.get("final_info", [None] * args.num_envs)
        infos_list = [
            (f if isinstance(f, dict) else {}) for f in (final_infos if isinstance(final_infos, list) else [final_infos])
        ][: args.num_envs]
        while len(infos_list) < args.num_envs:
            infos_list.append({})
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos_list)

        obs = next_obs

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten().float())
            old_val = q_network(data.observations).gather(1, data.actions.long()).squeeze()
            loss = F.mse_loss(td_target, old_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

        if global_step % args.target_network_frequency == 0 and global_step > args.learning_starts:
            for t_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                t_param.data.copy_(args.tau * q_param.data + (1.0 - args.tau) * t_param.data)

        sps = int(global_step / (time.time() - start_time))
        if global_step % 100 == 0:
            writer.add_scalar("charts/SPS", sps, global_step)

        if args.log_freq > 0 and global_step > 0 and global_step % args.log_freq < args.train_frequency:
            pct = 100.0 * global_step / args.total_timesteps
            mean_ret = f"{np.mean(recent_returns):.2f}" if recent_returns else "n/a"
            elapsed = time.time() - start_time
            eta = (elapsed / global_step) * (args.total_timesteps - global_step) if global_step > 0 else 0
            print(
                f"step={global_step}/{args.total_timesteps} ({pct:.1f}%) | "
                f"SPS={sps} | eps={epsilon:.3f} | return_mean={mean_ret} | eta={int(eta)}s"
            )

        if args.save_eval_freq > 0 and eval_env is not None and global_step > 0 and global_step % args.save_eval_freq < args.num_envs:
            eval_env.reset(seed=args.seed + 999)
            for _ in range(eval_env.max_steps):
                with torch.no_grad():
                    obs_t = torch.tensor(eval_env._get_obs(), device=device).unsqueeze(0)
                    q_vals = q_network(obs_t)
                    action = q_vals.argmax(dim=1).cpu().item()
                eval_env.step(action)
            img = eval_env.render()
            path = os.path.join(eval_save_path, f"eval_step_{global_step}.png")
            img.save(path)
            print(f"Saved evaluation drawing to {path}")

    envs.close()
    writer.close()

    if eval_env is not None:
        eval_env.reset(seed=args.seed)
        for _ in range(eval_env.max_steps):
            with torch.no_grad():
                obs_t = torch.tensor(eval_env._get_obs(), device=device).unsqueeze(0)
                q_vals = q_network(obs_t)
                action = q_vals.argmax(dim=1).cpu().item()
            eval_env.step(action)
        final_img = eval_env.render()
        out_path = os.path.join(run_dir, "final_drawing.png")
        final_img.save(out_path)
        print(f"Training done. Final drawing saved to {out_path}")
