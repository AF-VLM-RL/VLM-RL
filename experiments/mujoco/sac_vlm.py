# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
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
import wandb
import imageio
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer

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
    video_every: int = 1000
    """the frequency (measured in steps) of capturing videos"""
    run_id: Optional[str] = None
    """optional run identifier; defaults to current timestamp"""

    # VLM reward arguments
    vlm_model_type: str = "clip"
    """Type of VLM model to use. Options: 'clip', 'qwen'"""
    vlm_model_id: str = "openai/clip-vit-base-patch32"
    """CLIP model to use. Options: 'openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14'"""
    """Qwen model to use. Options: 'Qwen/Qwen2-VL-2B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct'"""
    vlm_goal: str = "an ant robot walking right stably"
    """The natural language goal for VLM reward shaping"""
    vlm_device: str = "auto"
    """Device to run the VLM on: 'auto', 'cuda', or 'cpu'"""
    vlm_n_frames: int = 4
    """Number of consecutive frames to concatenate before passing to CLIP"""
    vlm_frame_every: int = 4
    """Collect a frame into the buffer every K steps"""
    vlm_clip_every: int = 16
    """Run CLIP inference every M steps (must be a multiple of vlm_frame_every)"""

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
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
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


def make_env(env_id, seed, args):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = VLMRewardWrapper(
            env,
            model_id=args.vlm_model_id,
            text_goal=args.vlm_goal,
            device=args.vlm_device,
            n_frames=args.vlm_n_frames,
            frame_every=args.vlm_frame_every,
            clip_every=args.vlm_clip_every
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
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
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.vlm_device = resolve_vlm_device(args.vlm_device, args.cuda)

    if args.vlm_model_type == "clip":
        from src.wrappers_clip import VLMRewardWrapper
    elif args.vlm_model_type == "qwen":
        from src.wrappers_qwen import VLMRewardWrapper
    elif args.vlm_model_type == "xclip":
        from src.wrappers_xclip import VLMRewardWrapper

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_id = args.run_id if args.run_id else str(int(time.time()))
    run_name = f"{args.env_id}_{args.exp_name}_{run_id}"
    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    episode_count = 0
    episode_frames = []
    best_episode_return = -float("inf")
    best_episode_path = os.path.join(run_dir, "best_model.pt")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [ make_env(args.env_id, args.seed + i, args) for i in range(args.num_envs) ]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    assert isinstance(envs.single_observation_space, gym.spaces.Box), "only Box observation space is supported"
    

    # Logging setup
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Video setup
    if args.capture_video:
        video_dir = f"videos/{run_name}"
        video_fps = envs.envs[0].metadata.get("render_fps", 30)
        os.makedirs(video_dir, exist_ok=True)
    
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
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

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

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

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if args.capture_video:
            frame = envs.envs[0].unwrapped.render()
            episode_frames.append(frame)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            for i, finished in enumerate(infos["_episode"]):
                if finished:
                    ep_return = infos["episode"]["r"][i].item()
                    ep_length = infos["episode"]["l"][i].item()

                    print(f"global_step={global_step}, episode={episode_count}, episodic_return={ep_return:.3f}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)

                    if args.capture_video:
                        if ep_return > best_episode_return:
                            video_name = f"best_step-{global_step}_episode-{episode_count}"
                            video_path = f"videos/{run_name}/{video_name}.mp4"
                            imageio.mimsave(video_path, episode_frames, fps=video_fps)
                            print(f"    -> Best video saved: {video_path}")

                            if args.track:
                                wandb.log({video_name: wandb.Video(video_path, fps=video_fps, format="mp4")}, step=global_step)
                        elif episode_count % args.video_every == 0:
                            video_name = f"train_step-{global_step}_episode-{episode_count}"
                            video_path = f"videos/{run_name}/{video_name}.mp4"
                            imageio.mimsave(video_path, episode_frames, fps=video_fps)
                            print(f"    -> Train video saved: {video_path}")

                            if args.track:
                                wandb.log({video_name: wandb.Video(video_path, fps=video_fps, format="mp4")}, step=global_step)

                    if ep_return > best_episode_return:
                        best_episode_return = ep_return
                        torch.save({
                            "actor": actor.state_dict(),
                            "qf1": qf1.state_dict(),
                            "qf2": qf2.state_dict(),
                            "global_step": global_step,
                            "episodic_return": ep_return,
                        }, best_episode_path)
                        print(f"    -> New best: episodic_return={ep_return:.3f}")
                    
                    episode_frames = []
                    episode_count += 1

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                if "final_observation" in infos:
                    real_next_obs[idx] = infos["final_observation"][idx]
                elif "final_info" in infos and infos["final_info"][idx] is not None:
                    real_next_obs[idx] = infos["final_info"][idx]["terminal_observation"]


        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()

    if args.track:
        wandb.finish()