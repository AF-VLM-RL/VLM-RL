import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.wrappers import FrameBufferWrapper  # noqa: E402


@dataclass
class Args:
    # Environment / data collection
    env_id: str = "Walker2d-v4"
    num_envs: int = 8
    rollout_length: int = 512
    num_preference_batches: int = 200
    trajectories_per_batch: int = 32
    pairs_per_batch: int = 4

    # Reward model training
    reward_model_hidden_dim: int = 256
    reward_model_lr: float = 3e-4
    reward_model_weight_decay: float = 0.0
    reward_model_updates_per_batch: int = 64
    reward_clip: float = 5.0

    # VLM evaluator (teacher)
    vlm_model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    vlm_device: str = "cuda"
    vlm_goal: str = (
        "a 2D robot with a small central torso and two jointed legs but no arms, "
        "staying upright and walking steadily to the right over time without falling"
    )
    vlm_max_frames: int = 8
    vlm_layout: str = "grid"  # "strip" or "grid"
    vlm_grid_size: int = 3
    vlm_max_new_tokens: int = 64
    vlm_do_sample: bool = True
    vlm_temperature: float = 0.7
    vlm_top_p: float = 0.9

    # Logging / misc
    out_dir: str = "pref_logs"
    reward_model_path: str = "pref_logs/reward_model.pt"
    policy_path: Optional[str] = None
    save_pref_dataset: bool = False
    """If True, save sampled trajectories and VLM preference results for debugging."""
    pref_dataset_path: str = "pref_logs/pref_dataset_debug.npz"
    seed: int = 1


class Trajectory:
    def __init__(self, observations: np.ndarray, frames: List[np.ndarray]):
        self.observations = observations  # (T, obs_dim)
        self.frames = frames  # list length T (raw rgb frames or composites)


class TrajectoryBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage: List[Trajectory] = []

    def add(self, traj: Trajectory) -> None:
        self.storage.append(traj)
        if len(self.storage) > self.capacity:
            self.storage.pop(0)

    def sample_pairs(self, num_pairs: int) -> List[Tuple[Trajectory, Trajectory]]:
        assert len(self.storage) >= 2
        pairs: List[Tuple[Trajectory, Trajectory]] = []
        for _ in range(num_pairs):
            i, j = np.random.choice(len(self.storage), size=2, replace=False)
            pairs.append((self.storage[i], self.storage[j]))
        return pairs


class RewardModel(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class PreferenceDataset(Dataset):
    def __init__(self, pairs: List[Tuple[np.ndarray, np.ndarray]], labels: np.ndarray):
        self.pairs = pairs
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        obs_a, obs_b = self.pairs[idx]
        y = self.labels[idx]
        return (
            torch.from_numpy(obs_a).float(),  # (T_a, obs_dim) variable length
            torch.from_numpy(obs_b).float(),  # (T_b, obs_dim) variable length
            torch.tensor(y).float(),  # scalar
        )


def collate_variable_lengths(batch):
    """Return lists of variable-length tensors instead of stacking."""
    obs_a = [item[0] for item in batch]
    obs_b = [item[1] for item in batch]
    y = torch.stack([item[2] for item in batch])
    return obs_a, obs_b, y


class VLMPairwiseEvaluator:
    """VLM-based preference teacher operating on trajectory frame sequences."""

    def __init__(self, args: Args):
        from transformers import AutoModelForVision2Seq, AutoProcessor  # lazy import

        self.device = torch.device(args.vlm_device)
        self.processor = AutoProcessor.from_pretrained(args.vlm_model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            args.vlm_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="cuda" if self.device.type == "cuda" else None,
        )
        self.model.eval()
        self.goal = args.vlm_goal
        self.max_new_tokens = args.vlm_max_new_tokens
        self.do_sample = args.vlm_do_sample
        self.temperature = args.vlm_temperature
        self.top_p = args.vlm_top_p

    def _build_prompt(self) -> str:
        return (
            "You will see two short video strips from a 2D physics simulation, labelled A and B.\n"
            "Each strip shows a simple robot with a small central torso and two jointed legs but no arms.\n"
            f"Goal: {self.goal}\n"
            "Decide which strip better matches the goal, based ONLY on what you see.\n"
            "Consider uprightness (staying above the legs), forward progress to the right, stability of gait, "
            "and clear alternating leg movement (one leg stepping forward while the other supports).\n"
            "First, describe the key differences between A and B in 1-2 sentences.\n"
        )

    def preference_probs(
        self, composites_a: List[np.ndarray], composites_b: List[np.ndarray]
    ) -> np.ndarray:
        """Return P(A preferred to B) for each pair, shape (N,).

        Two-stage prompting:
          1. Ask the VLM to analyze differences between A and B.
          2. Feed that analysis back and ask for a discrete preference label:
             1 (A better), 0 (B better), -1 (unsure).
        """
        assert len(composites_a) == len(composites_b)
        n = len(composites_a)
        base_prompt = self._build_prompt()

        labels = np.full(n, -1.0, dtype=np.float32)

        for i in range(n):
            images = [composites_a[i], composites_b[i]]

            # Stage 1: analysis
            msg1 = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # A
                        {"type": "image"},  # B
                        {"type": "text", "text": base_prompt},
                    ],
                }
            ]
            text1 = self.processor.apply_chat_template(
                msg1, tokenize=False, add_generation_prompt=True
            )
            inputs1 = self.processor(
                text=[text1],
                images=[images],
                padding=True,
                return_tensors="pt",
            )
            inputs1 = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in inputs1.items()
            }
            with torch.no_grad():
                out_ids1 = self.model.generate(
                    **inputs1,
                    do_sample=True,
                    temperature=self.temperature if self.do_sample else None,
                    top_p=self.top_p if self.do_sample else None,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            in_len1 = inputs1["input_ids"].shape[1]
            analysis = self.processor.batch_decode(
                out_ids1[:, in_len1:], skip_special_tokens=True
            )[0].strip()

            # Stage 2: classification based on analysis only (no images this time).
            cls_prompt = (
                "You are given an analysis comparing two trajectories A and B:\n"
                f"{analysis}\n\n"
                "Based on this analysis, output a single integer label:\n"
                "1  if A is better than B for the goal.\n"
                "0  if B is better than A for the goal.\n"
                "-1 if you cannot decide or they are equally good/bad.\n"
                "Output only the integer, nothing else."
            )
            inputs2 = self.processor.tokenizer(
                [cls_prompt], return_tensors="pt", padding=True
            )
            inputs2 = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in inputs2.items()
            }
            with torch.no_grad():
                out_ids2 = self.model.generate(
                    **inputs2,
                    do_sample=False,
                    max_new_tokens=8,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            in_len2 = inputs2["input_ids"].shape[1]
            resp = self.processor.tokenizer.batch_decode(
                out_ids2[:, in_len2:], skip_special_tokens=True
            )[0].strip()

            m = re.search(r"-?1|0", resp)
            if not m:
                labels[i] = -1.0
            else:
                try:
                    val = int(m.group(0))
                except ValueError:
                    val = -1
                if val in (0, 1, -1):
                    labels[i] = float(val)
                else:
                    labels[i] = -1.0

        # Map 1 -> P=1, 0 -> P=0, -1 -> "unsure" (caller must filter out -1s).
        probs = np.where(labels >= 0.0, labels, -1.0).astype(np.float32)
        return probs


def collect_trajectories(args: Args) -> Tuple[TrajectoryBuffer, int]:
    """Collect trajectories using either a random policy or a loaded PPO policy."""
    envs = gym.vector.SyncVectorEnv(
        [
            lambda: FrameBufferWrapper(
                gym.wrappers.FlattenObservation(
                    gym.make(args.env_id, render_mode="rgb_array")
                ),
                max_frames=args.vlm_max_frames,
                layout=args.vlm_layout,
                grid_size=args.vlm_grid_size,
            )
            for _ in range(args.num_envs)
        ]
    )
    obs, _ = envs.reset(seed=args.seed)
    obs_dim = int(np.prod(envs.single_observation_space.shape))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = None
    if args.policy_path:
        # Import Agent definition from the PPO reward-model script style.
        from experiments.ppo_continuous_action_reward_model import Agent as PPOAgent  # type: ignore

        agent = PPOAgent(envs).to(device)
        state_dict = torch.load(args.policy_path, map_location=device)
        agent.load_state_dict(state_dict)
        agent.eval()

    buffer = TrajectoryBuffer(capacity=4 * args.trajectories_per_batch)
    steps_per_traj = args.rollout_length

    for _ in range(args.trajectories_per_batch):
        traj_obs: List[np.ndarray] = []
        traj_frames: List[np.ndarray] = []
        done_env = np.zeros(args.num_envs, dtype=bool)

        for _step in range(steps_per_traj):
            if agent is not None:
                with torch.no_grad():
                    obs_tensor = torch.Tensor(obs).to(device)
                    action, _, _, _ = agent.get_action_and_value(obs_tensor)
                    actions = action.cpu().numpy()
            else:
                actions = np.stack(
                    [envs.single_action_space.sample() for _ in range(args.num_envs)],
                    axis=0,
                )
            next_obs, _, terminations, truncations, infos = envs.step(actions)
            # For preference learning we only need observations and visuals.
            obs = next_obs
            done_env = np.logical_or(done_env, np.logical_or(terminations, truncations))

            # Take frames from env 0 as a representative trajectory.
            frame = envs.call("get_composite_image")[0]
            traj_obs.append(obs[0].copy())
            traj_frames.append(np.asarray(frame).copy())

            if done_env[0]:
                obs, _ = envs.reset()
                done_env[:] = False
                break

        if traj_obs:
            buffer.add(
                Trajectory(
                    observations=np.stack(traj_obs, axis=0),
                    frames=traj_frames,
                )
            )

    envs.close()
    return buffer, obs_dim


def train_reward_model_on_pairs(
    reward_model: RewardModel,
    optimizer: optim.Optimizer,
    pairs: List[Tuple[Trajectory, Trajectory]],
    pref_probs: np.ndarray,
    device: torch.device,
    updates: int,
) -> None:
    # Build dataset: for each pair (A,B), label = P(A ≻ B) from VLM.
    obs_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    labels = pref_probs
    for traj_a, traj_b in pairs:
        obs_pairs.append((traj_a.observations, traj_b.observations))

    # Filter out "unsure" labels (-1).
    labels = np.asarray(labels, dtype=np.float32)
    mask = (labels == 0.0) | (labels == 1.0)
    if not np.any(mask):
        print("No confident VLM preferences in this batch; skipping reward model update.")
        return
    labels = labels[mask]
    obs_pairs = [p for p, m in zip(obs_pairs, mask) if m]

    dataset = PreferenceDataset(obs_pairs, labels)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_variable_lengths,
    )

    bce = nn.BCEWithLogitsLoss()

    step_idx = 0
    for _ in range(updates):
        for obs_a, obs_b, y in loader:
            y = y.to(device)

            # Sum of rewards over each trajectory as "return" (variable lengths).
            r_a_sums = torch.stack([reward_model(x.to(device)).sum() for x in obs_a])
            r_b_sums = torch.stack([reward_model(x.to(device)).sum() for x in obs_b])

            logit = r_a_sums - r_b_sums  # P(A ≻ B) = sigmoid(r_a - r_b)
            loss = bce(logit, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_idx += 1
            if step_idx % 10 == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(logit).mean().item()
                print(
                    f"  [reward_model] step={step_idx}, "
                    f"batch_loss={loss.item():.4f}, "
                    f"mean_P(A>B)={probs:.3f}"
                )


def main():
    import tyro

    args = tyro.cli(Args)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    buffer, obs_dim = collect_trajectories(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model = RewardModel(obs_dim=obs_dim, hidden_dim=args.reward_model_hidden_dim).to(
        device
    )
    optimizer = optim.AdamW(
        reward_model.parameters(),
        lr=args.reward_model_lr,
        weight_decay=args.reward_model_weight_decay,
    )

    vlm = VLMPairwiseEvaluator(args)

    all_pairs_indices: List[Tuple[int, int]] = []
    all_pref_probs: List[np.ndarray] = []

    for batch_idx in range(args.num_preference_batches):
        pairs = buffer.sample_pairs(args.pairs_per_batch)

        # Prepare composites for VLM (last composite per trajectory).
        composites_a: List[np.ndarray] = []
        composites_b: List[np.ndarray] = []
        for traj_a, traj_b in pairs:
            composites_a.append(traj_a.frames[-1])
            composites_b.append(traj_b.frames[-1])

        pref_probs = vlm.preference_probs(composites_a, composites_b)
        all_pref_probs.append(pref_probs.copy())
        # Record which trajectories were paired (by index in buffer.storage)
        batch_indices: List[Tuple[int, int]] = []
        for traj_a, traj_b in pairs:
            ia = buffer.storage.index(traj_a)
            ib = buffer.storage.index(traj_b)
            batch_indices.append((ia, ib))
        all_pairs_indices.extend(batch_indices)

        train_reward_model_on_pairs(
            reward_model,
            optimizer,
            pairs,
            pref_probs,
            device,
            updates=args.reward_model_updates_per_batch,
        )

        print(
            f"[reward_model] batch {batch_idx+1}/{args.num_preference_batches} "
            f"VLM mean_pref={pref_probs.mean():.3f}, "
            f"std_pref={pref_probs.std():.3f}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = args.reward_model_path
    torch.save(reward_model.state_dict(), ckpt_path)
    print(f"Saved reward model to {ckpt_path}")

    if args.save_pref_dataset:
        # For testing/debugging: save trajectories (observations only) and VLM prefs.
        traj_obs_list = [traj.observations for traj in buffer.storage]
        # Store as a ragged array via object dtype.
        traj_obs_arr = np.array(traj_obs_list, dtype=object)
        pref_probs_arr = np.concatenate(all_pref_probs, axis=0) if all_pref_probs else np.array([])
        pairs_arr = np.array(all_pairs_indices, dtype=np.int32) if all_pairs_indices else np.empty((0, 2), dtype=np.int32)

        np.savez_compressed(
            args.pref_dataset_path,
            traj_observations=traj_obs_arr,
            pair_indices=pairs_arr,
            pref_probs=pref_probs_arr,
        )
        print(f"Saved preference debug dataset to {args.pref_dataset_path}")


if __name__ == "__main__":
    main()

