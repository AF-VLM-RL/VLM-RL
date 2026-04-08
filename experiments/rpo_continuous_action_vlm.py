# RPO (Robust Policy Optimization) with VLM frame-sequence rewards for continuous action environments.
# Same VLM reward pipeline as ppo_continuous_action_vlm.py; policy update uses perturbed Gaussian means
# per https://docs.cleanrl.dev/rl-algorithms/rpo/#rpo_continuous_actionpy
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Literal, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import sanitize_prompt_for_filename
from src.wrappers import (
    FrameBufferWrapper,
    score_rollout_composite_images,
    score_rollout_composite_images_clip,
    score_rollout_composite_images_clip_multi_goal,
    score_rollout_frame_sequences_clip,
    score_rollout_frame_sequences_clip_multi_goal,
    score_rollout_frame_sequences_video,
    score_rollout_frame_sequences_xclip,
    score_rollout_frame_sequences_xclip_multi_goal,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # VLM arguments
    vlm_backend: Literal["generate", "clip", "xclip"] = "generate"
    """`generate`: Qwen2-VL (slow, rich). `clip`: CLIP image–text similarity (fast, batched).
    `xclip`: X-CLIP video–text similarity (fast-ish, uses frame sequences as video)."""
    vlm_input_format: Literal["image", "video"] = "image"
    """For vlm_backend=generate: `image` uses strip/grid composite; `video` uses Qwen native video input."""
    vlm_video_fps: float = 8.0
    """Video fps hint passed to Qwen when vlm_input_format=video."""
    vlm_goal: str = "a cheetah running forward quickly"
    """Natural language goal for VLM reward shaping (used when vlm_goals is empty)."""
    vlm_goals: Optional[List[str]] = None
    """Multiple goals for CLIP weighted reward; overrides vlm_goal when set. E.g. ['upright', 'smooth gait', 'forward']."""
    vlm_goal_weights: Optional[List[float]] = None
    """Weights per goal when vlm_goals is set; normalized to sum to 1. Default: equal weights."""
    vlm_model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    """Hugging Face model id when vlm_backend=generate"""
    clip_model_name: str = "openai/clip-vit-base-patch32"
    """Hugging Face model id when vlm_backend=clip"""
    xclip_model_name: str = "microsoft/xclip-base-patch32"
    """Hugging Face model id when vlm_backend=xclip"""
    vlm_skip_frames: int = 16
    """(Deprecated) Kept for backwards compatibility."""
    vlm_rollout_chunk_size: int = 32
    """Max composite images per VLM batched call after rollout; <=0 = one batch over full rollout (may OOM)."""
    vlm_max_frames: int = 16
    """Max frames to buffer for sequence reward (16 frames ≈ ~0.5s for Walker2d control steps)"""
    vlm_reward_scale: float = 1.0
    """Scale factor for VLM reward (1-5 -> 0-1 by default)"""
    vlm_layout: str = "grid"
    """Frame layout: 'strip' (composite), 'grid' (composite), or 'pil' (per-frame CLIP encoding)."""
    vlm_grid_size: int = 4
    """Grid size when vlm_layout='grid' (4 -> 4x4 = 16 frames)"""
    vlm_frame_stride: int = 5
    """Store one frame every `vlm_frame_stride` env steps in the frame buffer."""
    vlm_device: Optional[str] = None
    """Device for VLM (cuda/cpu). None = same as RPO device. Use 'cpu' when GPU is crowded."""
    vlm_max_new_tokens: int = 64
    """Maximum number of new tokens Qwen can generate per VLM call."""
    vlm_do_sample: bool = True
    """Use stochastic decoding for VLM outputs."""
    vlm_temperature: float = 0.7
    """Sampling temperature for the VLM (only if vlm_do_sample=True)."""
    vlm_top_p: float = 0.9
    """Top-p nucleus sampling for the VLM (only if vlm_do_sample=True)."""
    vlm_num_samples: int = 3
    """Number of stochastic samples per VLM query; rewards are averaged."""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    rpo_alpha: float = 0.5
    """RPO: uniform noise on policy mean during policy loss (see CleanRL RPO continuous)."""

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, vlm_max_frames, vlm_layout, vlm_grid_size, vlm_frame_stride):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.FlattenObservation(env)

        env = FrameBufferWrapper(
            env,
            max_frames=vlm_max_frames,
            layout=vlm_layout,
            grid_size=vlm_grid_size,
            frame_stride=vlm_frame_stride,
        )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, rpo_alpha: float):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:
            z = torch.empty_like(action_mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            probs = Normal(action_mean + z, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    name_goal = (args.vlm_goals[0] if args.vlm_goals and len(args.vlm_goals) > 0 else args.vlm_goal)
    if args.vlm_goals and len(args.vlm_goals) > 1:
        name_goal = f"multi_{name_goal[:20]}"
    prompt_suffix = sanitize_prompt_for_filename(name_goal)
    run_name = f"{args.env_id}__{args.exp_name}__{prompt_suffix}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        if os.environ.get("SLURM_JOB_ID") and os.environ.get("WANDB_MODE") != "online":
            os.environ["WANDB_MODE"] = "offline"

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    vlm_device = (
        torch.device(args.vlm_device)
        if args.vlm_device
        else device
    )

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.vlm_max_frames,
                args.vlm_layout,
                args.vlm_grid_size,
                args.vlm_frame_stride,
            )
            for i in range(args.num_envs)
        ]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.rpo_alpha).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_composites: list = []
        rollout_frame_sequences: list = []
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, _reward_env, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            raw_composites = list(envs.call("get_composite_image"))
            raw_frame_sequences = list(envs.call("get_frame_sequence"))

            if "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info is not None:
                        if "terminal_composite" in info:
                            raw_composites[i] = info["terminal_composite"]
                        if "terminal_frames" in info:
                            raw_frame_sequences[i] = info["terminal_frames"]
                        if "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            rollout_composites.append(raw_composites)
            rollout_frame_sequences.append(raw_frame_sequences)

        flat_images = [
            rollout_composites[s][e]
            for s in range(args.num_steps)
            for e in range(args.num_envs)
        ]
        chunk_sz = args.vlm_rollout_chunk_size
        if chunk_sz <= 0:
            chunk_sz = max(1, len(flat_images))
        if args.vlm_backend == "clip":
            goals = args.vlm_goals if args.vlm_goals else [args.vlm_goal]
            if args.vlm_layout == "pil":
                flat_frame_sequences = [
                    rollout_frame_sequences[s][e]
                    for s in range(args.num_steps)
                    for e in range(args.num_envs)
                ]
                if len(goals) > 1:
                    weights = args.vlm_goal_weights if args.vlm_goal_weights else [1.0 / len(goals)] * len(goals)
                    r_flat = score_rollout_frame_sequences_clip_multi_goal(
                        flat_frame_sequences,
                        goals=goals,
                        goal_weights=weights,
                        clip_model_name=args.clip_model_name,
                        device=vlm_device,
                        reward_scale=args.vlm_reward_scale,
                        chunk_size=chunk_sz,
                    )
                else:
                    r_flat = score_rollout_frame_sequences_clip(
                        flat_frame_sequences,
                        goal=goals[0],
                        clip_model_name=args.clip_model_name,
                        device=vlm_device,
                        reward_scale=args.vlm_reward_scale,
                        chunk_size=chunk_sz,
                    )
            elif len(goals) > 1:
                weights = args.vlm_goal_weights if args.vlm_goal_weights else [1.0 / len(goals)] * len(goals)
                r_flat = score_rollout_composite_images_clip_multi_goal(
                    flat_images,
                    goals=goals,
                    goal_weights=weights,
                    clip_model_name=args.clip_model_name,
                    device=vlm_device,
                    reward_scale=args.vlm_reward_scale,
                    chunk_size=chunk_sz,
                )
            else:
                r_flat = score_rollout_composite_images_clip(
                    flat_images,
                    goal=goals[0],
                    clip_model_name=args.clip_model_name,
                    device=vlm_device,
                    reward_scale=args.vlm_reward_scale,
                    chunk_size=chunk_sz,
                )
        elif args.vlm_backend == "xclip":
            goals = args.vlm_goals if args.vlm_goals else [args.vlm_goal]
            flat_frame_sequences = [
                rollout_frame_sequences[s][e]
                for s in range(args.num_steps)
                for e in range(args.num_envs)
            ]
            if len(goals) > 1:
                weights = args.vlm_goal_weights if args.vlm_goal_weights else [1.0 / len(goals)] * len(goals)
                r_flat = score_rollout_frame_sequences_xclip_multi_goal(
                    flat_frame_sequences,
                    goals=goals,
                    goal_weights=weights,
                    xclip_model_name=args.xclip_model_name,
                    device=vlm_device,
                    reward_scale=args.vlm_reward_scale,
                    chunk_size=chunk_sz,
                )
            else:
                r_flat = score_rollout_frame_sequences_xclip(
                    flat_frame_sequences,
                    goal=goals[0],
                    xclip_model_name=args.xclip_model_name,
                    device=vlm_device,
                    reward_scale=args.vlm_reward_scale,
                    chunk_size=chunk_sz,
                )
        else:
            if args.vlm_input_format == "video":
                flat_frame_sequences = [
                    rollout_frame_sequences[s][e]
                    for s in range(args.num_steps)
                    for e in range(args.num_envs)
                ]
                r_flat = score_rollout_frame_sequences_video(
                    flat_frame_sequences,
                    goal=args.vlm_goal,
                    layout=args.vlm_layout,
                    model_name=args.vlm_model_name,
                    device=vlm_device,
                    reward_scale=args.vlm_reward_scale,
                    max_new_tokens=args.vlm_max_new_tokens,
                    vlm_do_sample=args.vlm_do_sample,
                    vlm_temperature=args.vlm_temperature,
                    vlm_top_p=args.vlm_top_p,
                    vlm_num_samples=args.vlm_num_samples,
                    chunk_size=chunk_sz,
                    video_fps=args.vlm_video_fps,
                )
            else:
                r_flat = score_rollout_composite_images(
                    flat_images,
                    goal=args.vlm_goal,
                    layout=args.vlm_layout,
                    model_name=args.vlm_model_name,
                    device=vlm_device,
                    reward_scale=args.vlm_reward_scale,
                    max_new_tokens=args.vlm_max_new_tokens,
                    vlm_do_sample=args.vlm_do_sample,
                    vlm_temperature=args.vlm_temperature,
                    vlm_top_p=args.vlm_top_p,
                    vlm_num_samples=args.vlm_num_samples,
                    chunk_size=chunk_sz,
                )
        rewards = torch.from_numpy(r_flat.reshape(args.num_steps, args.num_envs)).float().to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
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
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

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
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
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

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
