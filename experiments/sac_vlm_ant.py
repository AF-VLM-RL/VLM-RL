# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
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
from src.utils import sanitize_prompt_for_filename
from src.wrappers import VLMRewardWrapper


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
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # VLM reward arguments
    vlm_goal: str = "an ant robot walking right stably"
    """The natural language goal for VLM reward shaping"""
    vlm_device: str = "auto"
    """Device to run the VLM on: 'auto', 'cuda', or 'cpu'"""
    vlm_skip_frames: int = 16
    """Recompute VLM reward every N env steps"""

    # Algorithm specific arguments
    env_id: str = "Ant-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def resolve_vlm_device(vlm_device, use_cuda):
    if vlm_device in ("cuda", "cpu"):
        return vlm_device
    if vlm_device == "auto":
        if torch.cuda.is_available() and use_cuda:
            return "cuda"
        return "cpu"
    raise ValueError("vlm_device must be one of: 'auto', 'cuda', 'cpu'")


def make_env(env_id, seed, idx, capture_video, run_name, vlm_goal, vlm_device, vlm_skip_frames):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, "videos/{0}".format(run_name))
        env = VLMRewardWrapper(
            env,
            text_goal=vlm_goal,
            device=vlm_device,
            skip_frames=vlm_skip_frames,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))

        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.vlm_device = resolve_vlm_device(args.vlm_device, args.cuda)

    prompt_slug = sanitize_prompt_for_filename(args.vlm_goal)
    run_name = "{0}__{1}__{2}__{3}__{4}".format(
        args.env_id,
        args.exp_name,
        prompt_slug,
        args.seed,
        int(time.time()),
    )
    run_dir = os.path.join(_PROJECT_ROOT, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    if args.track:
        import wandb  # pyright: ignore[reportMissingImports]

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{0}".format(
            "\n".join(
                ["|{0}|{1}|".format(key, value) for key, value in vars(args).items()]
            )
        ),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                i,
                args.capture_video,
                run_name,
                args.vlm_goal,
                args.vlm_device,
                args.vlm_skip_frames,
            )
            for i in range(args.num_envs)
        ]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    assert isinstance(envs.single_observation_space, gym.spaces.Box), "only Box observation space is supported"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    last_actor_loss = None
    last_alpha_loss = None

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)],
                dtype=np.float32,
            )
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions, _, _ = actor.get_action(obs_tensor)
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    print(
                        "global_step={0}, episodic_return={1}".format(
                            global_step, info["episode"]["r"]
                        )
                    )
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break
        elif "episode" in infos:
            for i, done in enumerate(infos.get("_episode", [])):
                if done:
                    ep_return = infos["episode"]["r"][i].item()
                    ep_length = infos["episode"]["l"][i].item()
                    print(
                        "global_step={0}, episodic_return={1:.3f}".format(
                            global_step, ep_return
                        )
                    )
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)

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

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (
                    1.0 - data.dones.flatten()
                ) * args.gamma * min_qf_next_target.view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)

            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    last_actor_loss = actor_loss.item()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)

                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()

                        alpha = log_alpha.exp().item()
                        last_alpha_loss = alpha_loss.item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

                if last_actor_loss is not None:
                    writer.add_scalar("losses/actor_loss", last_actor_loss, global_step)

                if args.autotune and last_alpha_loss is not None:
                    writer.add_scalar("losses/alpha_loss", last_alpha_loss, global_step)

    envs.close()
    writer.close()