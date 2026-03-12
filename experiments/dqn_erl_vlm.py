# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # transformers<4.57 fallback
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

import sys
# Ensure local project modules resolve on cluster runs.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CLEANRL_ROOT = os.path.join(_PROJECT_ROOT, "cleanrl")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _CLEANRL_ROOT not in sys.path:
    sys.path.insert(0, _CLEANRL_ROOT)

from cleanrl_utils.buffers import ReplayBuffer  # pyright: ignore[reportMissingImports]
from src.utils import sanitize_prompt_for_filename

if not os.environ.get("XDG_RUNTIME_DIR"):
    _xdg_runtime_dir = f"/tmp/xdg-runtime-{os.getuid()}"
    os.makedirs(_xdg_runtime_dir, mode=0o700, exist_ok=True)
    os.environ["XDG_RUNTIME_DIR"] = _xdg_runtime_dir


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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # ERL-VLM arguments
    vlm_query_interval: int = 10
    """How often to query the VLM in timesteps"""
    rm_train_epochs: int = 3
    """Epochs to train the Reward Model"""
    rm_lr: float = 1e-4
    """Learning rate for the Reward Model"""
    rating_batch_size: int = 64
    """Batch size for training the RM"""
    vlm_goal: str = "a pole balancing upright on a cart"
    """Natural language goal used for VLM rating prompts"""
    vlm_model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    """Hugging Face model id for the generative VLM rater"""
    vlm_do_sample: bool = True
    """Enable stochastic decoding for VLM generation"""
    vlm_temperature: float = 0.2
    """Sampling temperature for VLM generation"""
    vlm_top_p: float = 0.9
    """Top-p sampling for VLM generation"""
    vlm_max_new_tokens: int = 40
    """Max generated tokens for VLM output"""
    vlm_collapse_threshold: float = 0.8
    """Re-query batch when one rating dominates this fraction"""
    initial_annotation_target: int = 2048
    """Number of VLM-rated samples to gather before Q-learning updates start"""
    initial_rm_pretrain_epochs: int = 20
    """RM pretraining epochs after initial annotation collection"""
    warmstart_dataset_path: str = "runs/warmstart_datasets/dqn_erl_vlm_cartpole.npz"
    """Relative or absolute path for persisted ERL warm-start annotations"""
    warmstart_autosave_interval: int = 1000
    """Save annotation cache every N steps when new labels were added (0 disables)"""
    run_dir_base: Optional[str] = None
    """Base directory for runs (TensorBoard, models). If unset, uses VLM_RL_RUN_DIR, else PROJECT_DIR/runs when on cluster, else project runs/."""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.AddRenderObservation(env, render_only=True)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_dim = int(np.array(env.single_observation_space.shape).prod())
        self.network = nn.Sequential(
            nn.Linear(in_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = x.view(x.shape[0], -1)
        return self.network(x)


class RewardModel(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError(f"RewardModel expects image observations [H, W, C], got {obs_shape}.")
        h, w, c = obs_shape
        if c not in (1, 3):
            raise ValueError(f"RewardModel expects channels in {{1, 3}}, got {c}.")

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = int(np.prod(self.cnn(dummy).shape[1:]))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        return self.head(self.cnn(x)).squeeze(-1)


class ERLDataset:
    def __init__(self):
        self.buckets: defaultdict[int, list[np.ndarray]] = defaultdict(list)

    def add(self, obs: np.ndarray, rating: int):
        if 1 <= rating <= 5:
            self.buckets[int(rating)].append(np.array(obs, copy=True))

    def __len__(self):
        return sum(len(v) for v in self.buckets.values())

    def available_ratings(self):
        return [rating for rating in range(1, 6) if len(self.buckets[rating]) > 0]

    def sample_stratified(self, batch_size: int):
        ratings = self.available_ratings()
        if not ratings:
            raise ValueError("Cannot sample from an empty ERLDataset.")

        per_rating = max(1, batch_size // len(ratings))
        samples, labels = [], []
        for rating in ratings:
            bucket = self.buckets[rating]
            for _ in range(per_rating):
                samples.append(random.choice(bucket))
                labels.append(float(rating))

        while len(samples) < batch_size:
            rating = random.choice(ratings)
            samples.append(random.choice(self.buckets[rating]))
            labels.append(float(rating))

        if len(samples) > batch_size:
            inds = random.sample(range(len(samples)), batch_size)
            samples = [samples[i] for i in inds]
            labels = [labels[i] for i in inds]

        return np.stack(samples, axis=0), np.array(labels, dtype=np.float32)

    def save_npz(self, path: str) -> int:
        all_obs = []
        all_ratings = []
        for rating in range(1, 6):
            bucket = self.buckets[rating]
            all_obs.extend(bucket)
            all_ratings.extend([rating] * len(bucket))

        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        if len(all_obs) == 0:
            np.savez_compressed(
                path,
                observations=np.empty((0,), dtype=np.uint8),
                ratings=np.empty((0,), dtype=np.int64),
            )
            return 0

        obs_np = np.stack(all_obs, axis=0)
        ratings_np = np.asarray(all_ratings, dtype=np.int64)
        np.savez_compressed(path, observations=obs_np, ratings=ratings_np)
        return int(ratings_np.shape[0])

    def load_npz(self, path: str, max_samples: Optional[int] = None) -> int:
        if not os.path.exists(path):
            return 0

        data = np.load(path)
        if "observations" not in data or "ratings" not in data:
            return 0

        observations = data["observations"]
        ratings = data["ratings"]
        total = min(len(observations), len(ratings))
        if max_samples is not None:
            total = min(total, max_samples)

        loaded = 0
        for i in range(total):
            rating = int(ratings[i])
            if 1 <= rating <= 5:
                self.buckets[rating].append(np.array(observations[i], copy=True))
                loaded += 1
        return loaded


def train_reward_model(
    reward_model: nn.Module,
    rm_optimizer: optim.Optimizer,
    rating_dataset: ERLDataset,
    rm_criterion: nn.Module,
    device: torch.device,
    epochs: int,
    batch_size: int,
):
    if len(rating_dataset) == 0:
        return 0.0

    reward_model.train()
    batches_per_epoch = max(1, len(rating_dataset) // batch_size)
    running_loss = 0.0
    updates = 0

    for _ in range(epochs):
        for _ in range(batches_per_epoch):
            batch_obs_np, batch_ratings_np = rating_dataset.sample_stratified(batch_size)
            batch_obs = torch.tensor(batch_obs_np, dtype=torch.float32, device=device)
            batch_ratings = torch.tensor(batch_ratings_np, dtype=torch.float32, device=device)

            pred = reward_model(batch_obs)
            rm_loss = rm_criterion(pred, batch_ratings)

            rm_optimizer.zero_grad()
            rm_loss.backward()
            rm_optimizer.step()

            running_loss += rm_loss.item()
            updates += 1

    return running_loss / max(1, updates)


# Lazy VLM cache so we do not reload the generative model every query.
_VLM_MODEL: Optional[nn.Module] = None
_VLM_PROCESSOR: Optional[AutoProcessor] = None
_VLM_DEVICE: Optional[torch.device] = None
_VLM_MODEL_NAME: Optional[str] = None

_NUMBER_WORD_TO_INT = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def _init_vlm(model_name: str):
    global _VLM_MODEL, _VLM_PROCESSOR, _VLM_DEVICE, _VLM_MODEL_NAME
    if _VLM_MODEL is not None and _VLM_MODEL_NAME == model_name:
        return

    _VLM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if _VLM_DEVICE.type == "cuda" else torch.float32
    _VLM_PROCESSOR = AutoProcessor.from_pretrained(model_name)
    _VLM_MODEL = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype).to(_VLM_DEVICE)
    _VLM_MODEL.eval()
    _VLM_MODEL_NAME = model_name


def _parse_rating(text: str) -> int:
    text_l = text.lower()
    tagged_match = re.search(r"(?:rating|score)\s*[:=]?\s*([1-5])\b", text_l)
    if tagged_match:
        return int(tagged_match.group(1))

    digit_match = re.search(r"\b([1-5])\b", text_l)
    if digit_match:
        return int(digit_match.group(1))

    for word, value in _NUMBER_WORD_TO_INT.items():
        if re.search(rf"\b{word}\b", text_l):
            return value
    return 3


def _extract_raw_score(text: str) -> float:
    text_l = text.lower()
    tagged_decimal = re.search(r"(?:rating|score)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text_l)
    if tagged_decimal:
        value = float(tagged_decimal.group(1))
        if 1.0 <= value <= 5.0:
            return value
        if 0.0 <= value <= 1.0:
            return 1.0 + 4.0 * value
        if 0.0 <= value <= 100.0:
            return 1.0 + 4.0 * (value / 100.0)

    decimal_match = re.search(r"(-?\d+(?:\.\d+)?)", text_l)
    if decimal_match:
        value = float(decimal_match.group(1))
        if 1.0 <= value <= 5.0:
            return value
        if 0.0 <= value <= 1.0:
            return 1.0 + 4.0 * value
        if 0.0 <= value <= 100.0:
            return 1.0 + 4.0 * (value / 100.0)
    return float(_parse_rating(text))


def _build_rating_prompt(goal: str, rubric_variant: int = 0) -> str:
    if rubric_variant == 0:
        return (
            "You are evaluating a CartPole frame for reward shaping.\n"
            f"Goal: {goal}\n"
            "Assign a score from 1 to 5 using this rubric:\n"
            "1 = pole has clearly fallen / near horizontal.\n"
            "2 = pole strongly tilted and unstable.\n"
            "3 = pole partly upright with visible tilt.\n"
            "4 = pole mostly upright with small tilt.\n"
            "5 = pole upright and stable above the cart.\n"
            "First, describe what you see in one short sentence.\n"
            "Then output the score.\n"
            "Use this exact format:\n"
            "Description: <short description>\n"
            "Rating: <1-5>"
        )
    return (
        "Rate this CartPole frame for balancing quality.\n"
        f"Objective: {goal}\n"
        "Use strict labels: 1 very poor, 2 poor, 3 moderate, 4 good, 5 excellent.\n"
        "Output exactly two lines:\n"
        "Description: <short sentence>\n"
        "Rating: <1-5>"
    )


def query_vlm_for_rating(
    obs: np.ndarray,
    goal: str,
    model_name: Optional[str] = None,
    do_sample: bool = True,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_new_tokens: int = 40,
    rubric_variant: int = 0,
):
    model_name = model_name or os.getenv("ERL_VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
    _init_vlm(model_name)
    assert _VLM_MODEL is not None and _VLM_PROCESSOR is not None and _VLM_DEVICE is not None

    frame = np.asarray(obs)
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] != 3:
        raise ValueError(f"Expected RGB observation [H, W, 3], got {frame.shape}.")

    image = Image.fromarray(frame.astype(np.uint8))
    prompt = _build_rating_prompt(goal, rubric_variant=rubric_variant)

    if hasattr(_VLM_PROCESSOR, "apply_chat_template"):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = _VLM_PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _VLM_PROCESSOR(text=[text], images=[image], return_tensors="pt")
    else:
        inputs = _VLM_PROCESSOR(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(_VLM_DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _VLM_MODEL.generate(
            **inputs,
            do_sample=do_sample,
            temperature=max(1e-5, temperature) if do_sample else None,
            top_p=top_p if do_sample else None,
            max_new_tokens=max_new_tokens,
        )

    if "input_ids" in inputs:
        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = output_ids[:, prompt_len:]
        generated = _VLM_PROCESSOR.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()
    else:
        generated = _VLM_PROCESSOR.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    raw_score = float(np.clip(_extract_raw_score(generated), 1.0, 5.0))
    rating = int(np.clip(_parse_rating(generated), 1, 5))
    return rating, raw_score, generated


def annotate_observation_batch(observations: list[np.ndarray], args: Args):
    discrete_ratings = []
    for obs in observations:
        rating, _, _ = query_vlm_for_rating(
            obs=obs,
            goal=args.vlm_goal,
            model_name=args.vlm_model_name,
            do_sample=args.vlm_do_sample,
            temperature=args.vlm_temperature,
            top_p=args.vlm_top_p,
            max_new_tokens=args.vlm_max_new_tokens,
            rubric_variant=0,
        )
        discrete_ratings.append(rating)

    if not discrete_ratings:
        return []

    dominant_fraction = Counter(discrete_ratings).most_common(1)[0][1] / len(discrete_ratings)
    if dominant_fraction >= args.vlm_collapse_threshold and len(discrete_ratings) > 1:
        discrete_ratings = []
        for obs in observations:
            rating, _, _ = query_vlm_for_rating(
                obs=obs,
                goal=args.vlm_goal,
                model_name=args.vlm_model_name,
                do_sample=True,
                temperature=max(0.3, args.vlm_temperature),
                top_p=args.vlm_top_p,
                max_new_tokens=args.vlm_max_new_tokens,
                rubric_variant=1,
            )
            discrete_ratings.append(rating)
    return discrete_ratings


def _get_run_dir_base(run_dir_base: Optional[str], project_root: str) -> str:
    """Resolve runs base: explicit arg > VLM_RL_RUN_DIR > PROJECT_DIR/runs > project runs/."""
    if run_dir_base is not None:
        return run_dir_base
    if base := os.environ.get("VLM_RL_RUN_DIR"):
        return base
    if project_dir := os.environ.get("PROJECT_DIR"):
        return os.path.join(project_dir, "runs")
    return os.path.join(project_root, "runs")


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    prompt_slug = sanitize_prompt_for_filename(args.vlm_goal)
    run_name = f"{args.env_id}__{args.exp_name}__{prompt_slug}__{args.seed}__{int(time.time())}"
    run_dir_base = _get_run_dir_base(args.run_dir_base, _PROJECT_ROOT)
    run_dir = os.path.join(run_dir_base, run_name)
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    reward_model = RewardModel(envs.single_observation_space.shape).to(device)
    rm_optimizer = optim.Adam(reward_model.parameters(), lr=args.rm_lr)
    rating_dataset = ERLDataset()
    rm_criterion = nn.L1Loss()
    warmstart_dataset_path = (
        args.warmstart_dataset_path
        if os.path.isabs(args.warmstart_dataset_path)
        else os.path.join(_PROJECT_ROOT, args.warmstart_dataset_path)
    )
    loaded_annotations = rating_dataset.load_npz(warmstart_dataset_path)
    if loaded_annotations > 0:
        print(
            f"Loaded {loaded_annotations} warm-start annotations from {warmstart_dataset_path}. "
            f"Current dataset size={len(rating_dataset)}"
        )

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    # Warm-start: gather annotated data before Q-learning updates start.
    initial_annotations = 0
    remaining_warmstart_target = max(0, args.initial_annotation_target - len(rating_dataset))
    if remaining_warmstart_target > 0:
        print(
            "Starting initial annotation warmup for "
            f"{remaining_warmstart_target} additional samples "
            f"(target={args.initial_annotation_target}, loaded={loaded_annotations})..."
        )
        collected_ratings = []
        while initial_annotations < remaining_warmstart_target:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            next_obs, _, terminations, truncations, _ = envs.step(actions)
            remaining = remaining_warmstart_target - initial_annotations
            obs_batch = next_obs[:remaining]
            ratings = annotate_observation_batch([o for o in obs_batch], args)
            for sampled, rating in zip(obs_batch, ratings):
                rating_dataset.add(sampled, rating)
                collected_ratings.append(rating)
            initial_annotations += len(obs_batch)
            obs = next_obs

            if np.any(np.logical_or(terminations, truncations)):
                obs, _ = envs.reset()

            if initial_annotations % max(64, args.rating_batch_size) == 0:
                print(f"Warmup annotations collected: {initial_annotations}/{remaining_warmstart_target}")

        initial_mean_rating = float(np.mean(collected_ratings)) if collected_ratings else 0.0
    else:
        initial_mean_rating = 0.0
        print(
            "Warm-start target already satisfied by cached dataset: "
            f"{len(rating_dataset)}/{args.initial_annotation_target}"
        )

    if len(rating_dataset) > 0 and args.initial_rm_pretrain_epochs > 0:
        pretrain_loss = train_reward_model(
            reward_model=reward_model,
            rm_optimizer=rm_optimizer,
            rating_dataset=rating_dataset,
            rm_criterion=rm_criterion,
            device=device,
            epochs=args.initial_rm_pretrain_epochs,
            batch_size=args.rating_batch_size,
        )
        print(
            "Initial RM pretraining complete: "
            f"dataset={len(rating_dataset)} mean_rating={initial_mean_rating:.2f} pretrain_loss={pretrain_loss:.4f}"
        )
    else:
        pretrain_loss = 0.0

    writer.add_scalar("erl/initial_loaded_annotations", loaded_annotations, 0)
    writer.add_scalar("erl/initial_annotations", initial_annotations, 0)
    writer.add_scalar("erl/initial_mean_vlm_rating", initial_mean_rating, 0)
    writer.add_scalar("erl/initial_pretrain_loss", pretrain_loss, 0)

    cached_size = rating_dataset.save_npz(warmstart_dataset_path)
    print(f"Saved warm-start annotation cache ({cached_size} samples) to {warmstart_dataset_path}")

    recent_observations: list[np.ndarray] = []

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.tensor(obs, dtype=torch.float32, device=device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, _, terminations, truncations, infos = envs.step(actions)
        recent_observations.extend(next_obs.copy())
        if len(recent_observations) > args.rating_batch_size * 20:
            recent_observations = recent_observations[-args.rating_batch_size * 20 :]

        # Phase A: periodic VLM annotation from recent observations.
        annotations_added = 0
        mean_vlm_rating = 0.0
        if global_step % max(1, args.vlm_query_interval) == 0 and len(recent_observations) > 0:
            num_to_rate = min(args.rating_batch_size, len(recent_observations))
            sampled_obs = random.sample(recent_observations, num_to_rate)
            sampled_ratings = annotate_observation_batch(sampled_obs, args)
            for sampled, rating in zip(sampled_obs, sampled_ratings):
                rating_dataset.add(sampled, rating)
            annotations_added = len(sampled_ratings)
            mean_vlm_rating = float(np.mean(sampled_ratings)) if sampled_ratings else 0.0
            if args.warmstart_autosave_interval > 0 and global_step % args.warmstart_autosave_interval == 0:
                cached_size = rating_dataset.save_npz(warmstart_dataset_path)
                print(f"Autosaved warm-start cache ({cached_size} samples) at step {global_step}")

        # Phase B: train distilled reward model periodically.
        rm_loss_value = 0.0
        if global_step > args.learning_starts and global_step % args.train_frequency == 0 and len(rating_dataset) > 0:
            rm_loss_value = train_reward_model(
                reward_model=reward_model,
                rm_optimizer=rm_optimizer,
                rating_dataset=rating_dataset,
                rm_criterion=rm_criterion,
                device=device,
                epochs=args.rm_train_epochs,
                batch_size=args.rating_batch_size,
            )

        # Phase C: use RM reward for replay storage.
        with torch.no_grad():
            rm_rewards = reward_model(torch.tensor(next_obs, dtype=torch.float32, device=device))
        rewards = rm_rewards.detach().cpu().numpy()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # --- NEW: Gymnasium >= 0.28 VectorEnv format ---
        elif "episode" in infos:
            # '_episode' is a boolean array indicating which envs just terminated
            for i, done in enumerate(infos.get("_episode", [])):
                if done:
                    ep_return = infos["episode"]["r"][i].item() # .item() extracts the float
                    ep_length = infos["episode"]["l"][i].item()
                    print(f"global_step={global_step}, episodic_return={ep_return:.3f}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
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

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("erl/rm_loss", rm_loss_value, global_step)
                    writer.add_scalar("erl/dataset_size", len(rating_dataset), global_step)
                    writer.add_scalar("erl/annotations_added", annotations_added, global_step)
                    writer.add_scalar("erl/mean_vlm_rating", mean_vlm_rating, global_step)
                    writer.add_scalar("erl/rm_reward_mean", float(np.mean(rewards)), global_step)
                    writer.add_scalar("erl/rm_reward_std", float(np.std(rewards)), global_step)
                    for rating in range(1, 6):
                        writer.add_scalar(f"erl/rating_bucket_{rating}", len(rating_dataset.buckets[rating]), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    final_cached_size = rating_dataset.save_npz(warmstart_dataset_path)
    print(f"Saved final warm-start annotation cache ({final_cached_size} samples) to {warmstart_dataset_path}")

    if args.save_model:
        model_path = os.path.join(run_dir, f"{args.exp_name}.cleanrl_model")
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate  # pyright: ignore[reportMissingImports]

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub  # pyright: ignore[reportMissingImports]

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", run_dir, f"videos/{run_name}-eval")

    envs.close()
    writer.close()
