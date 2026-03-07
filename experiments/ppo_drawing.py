"""
CleanRL-style PPO for the hand-drawing circle environment.

Uses a CNN policy for 2-channel (canvas + cursor) image observations and
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
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
    capture_video: bool = False
    save_eval_freq: int = 10000
    """Save evaluation drawing every N steps (0 disables)"""
    log_freq: int = 10
    """Print progress every N PPO iterations (0 disables periodic logging)"""
    eval_save_path: str = "runs/ppo_drawing_eval"
    """Directory to save evaluation drawings (relative to project root)"""
    run_dir_base: Optional[str] = None
    """Base directory for runs. If unset, uses VLM_RL_RUN_DIR, else PROJECT_DIR/runs when on cluster, else project runs/."""

    total_timesteps: int = 500_000
    learning_rate: float = 1e-4
    num_envs: int = 4
    """Number of parallel envs. Each DrawingEnv loads its own CLIP; 4 envs ≈4× VRAM for reward models."""
    num_steps: int = 512
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class CnnAgent(nn.Module):
    """CNN policy for (C, H, W) image observations."""

    def __init__(self, envs):
        super().__init__()
        c, h, w = envs.single_observation_space.shape
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

        self.critic = nn.Sequential(
            layer_init(nn.Linear(flat_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(flat_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        x = x.float() / 255.0
        feat = self.cnn(x)
        return self.critic(feat)

    def get_action(self, x, deterministic=False):
        x = x.float() / 255.0
        feat = self.cnn(x)
        logits = self.actor(feat)
        if deterministic:
            return logits.argmax(dim=-1)
        return Categorical(logits=logits).sample()

    def get_action_and_value(self, x, action=None):
        x = x.float() / 255.0
        feat = self.cnn(x)
        logits = self.actor(feat)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(feat)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

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

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    agent = CnnAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_shape = envs.single_observation_space.shape
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.long)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    eval_env = DrawingEnv() if args.save_eval_freq > 0 else None
    if eval_env and args.save_eval_freq > 0:
        os.makedirs(eval_save_path, exist_ok=True)

    recent_returns = []
    max_recent = 20
    best_episodic_return = -float("inf")
    best_model_path = os.path.join(run_dir, "best_model.pt")

    print(f"Starting training: {args.num_iterations} iterations, {args.total_timesteps} total steps")
    print(f"Progress will be logged every {args.log_freq} iterations")

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.tensor(next_obs, device=device)
            next_done = torch.tensor(next_done, device=device)

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
                        if ep_ret > best_episodic_return:
                            best_episodic_return = ep_ret
                            torch.save(agent.state_dict(), best_model_path)
                            print(f"--> New best model saved (return={best_episodic_return:.2f})")
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
                        if ep_ret > best_episodic_return:
                            best_episodic_return = ep_ret
                            torch.save(agent.state_dict(), best_model_path)
                            print(f"--> New best model saved (return={best_episodic_return:.2f})")

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        var_y = np.var(b_returns.cpu().numpy())
        explained_var = np.nan if var_y == 0 else 1 - np.var((b_returns - b_values).cpu().numpy()) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        if args.log_freq > 0 and iteration % args.log_freq == 0:
            pct = 100.0 * global_step / args.total_timesteps
            mean_ret = f"{np.mean(recent_returns):.2f}" if recent_returns else "n/a"
            elapsed = time.time() - start_time
            eta = (elapsed / global_step) * (args.total_timesteps - global_step) if global_step > 0 else 0
            print(
                f"[{iteration}/{args.num_iterations}] step={global_step}/{args.total_timesteps} ({pct:.1f}%) | "
                f"SPS={sps} | return_mean={mean_ret} | pg_loss={pg_loss.item():.3f} v_loss={v_loss.item():.3f} | "
                f"eta={int(eta)}s"
            )

        if args.save_eval_freq > 0 and eval_env is not None and global_step % args.save_eval_freq < args.batch_size:
            eval_env.reset(seed=args.seed + 999)
            for _ in range(eval_env.max_steps):
                with torch.no_grad():
                    obs_t = torch.tensor(eval_env._get_obs(), device=device).unsqueeze(0)
                    action = agent.get_action(obs_t, deterministic=True)
                eval_env.step(action.cpu().item())
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
                action = agent.get_action(obs_t, deterministic=True)
            eval_env.step(action.cpu().item())
        final_img = eval_env.render()
        out_path = os.path.join(run_dir, "final_drawing.png")
        final_img.save(out_path)
        print(f"Training done. Final drawing saved to {out_path}")
