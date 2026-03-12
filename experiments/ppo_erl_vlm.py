"""PPO with ERL-VLM style reward modeling.

This script decouples VLM querying from the active PPO loop:
1) Query VLM periodically for discrete ratings (1-5) on sampled observations.
2) Train a lightweight reward model (RM) on those ratings.
3) Use RM predictions as rollout rewards for PPO.
"""

import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForVision2Seq, AutoProcessor

from src.utils import sanitize_prompt_for_filename


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

    # ERL-VLM arguments
    vlm_query_interval: int = 10
    """How often to query the VLM in iterations"""
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
    """Enable stochastic decoding for VLM rating generation"""
    vlm_temperature: float = 0.2
    """Sampling temperature for VLM generation"""
    vlm_top_p: float = 0.9
    """Top-p sampling for VLM generation"""
    vlm_max_new_tokens: int = 40
    """Max generated tokens for VLM rating output"""
    vlm_collapse_threshold: float = 0.8
    """Re-query batch when one rating dominates this fraction"""
    initial_annotation_target: int = 2048
    """Number of VLM-rated samples to gather before PPO updates start"""
    initial_rm_pretrain_epochs: int = 20
    """RM pretraining epochs after initial annotation collection"""
    run_dir_base: Optional[str] = None
    """Base directory for runs (TensorBoard, models). If unset, uses VLM_RL_RUN_DIR, else PROJECT_DIR/runs when on cluster, else project runs/."""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles clipped loss for value function"""
    ent_coef: float = 0.01
    """coefficient of entropy"""
    vf_coef: float = 0.5
    """coefficient of value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.AddRenderObservation(env, render_only=True)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x % 50 == 0)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# Lazy VLM cache so we do not reload the generative model every query.
_VLM_MODEL: Optional[AutoModelForVision2Seq] = None
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
    _VLM_MODEL = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=dtype).to(_VLM_DEVICE)
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

    # Conservative fallback so noisy generations do not break training.
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
        # Handle common formats:
        # 1..5 directly, 0..1 normalized, or 0..100 percentage-like scale.
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
    max_new_tokens: int = 12,
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
    raw_scores = []

    for obs in observations:
        rating, raw_score, _ = query_vlm_for_rating(
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
        raw_scores.append(raw_score)

    if not discrete_ratings:
        return []

    dominant_fraction = Counter(discrete_ratings).most_common(1)[0][1] / len(discrete_ratings)
    if dominant_fraction >= args.vlm_collapse_threshold and len(discrete_ratings) > 1:
        # Re-query with alternate rubric wording to break collapsed responses.
        discrete_ratings = []
        raw_scores = []
        for obs in observations:
            rating, raw_score, _ = query_vlm_for_rating(
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
            raw_scores.append(raw_score)

    return discrete_ratings


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
        samples = []
        labels = []

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

        obs_batch = np.stack(samples, axis=0)
        rating_batch = np.array(labels, dtype=np.float32)
        return obs_batch, rating_batch


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
        features = self.cnn(x)
        out = self.head(features)
        return out.squeeze(-1)


def train_reward_model(
    reward_model: nn.Module,
    rm_optimizer: optim.Optimizer,
    rating_dataset: "ERLDataset",
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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def _flatten_obs(self, x: torch.Tensor):
        return x.view(x.shape[0], -1)

    def get_value(self, x):
        return self.critic(self._flatten_obs(x))

    def get_action_and_value(self, x, action=None):
        logits = self.actor(self._flatten_obs(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(self._flatten_obs(x))


def _get_run_dir_base(run_dir_base: Optional[str]) -> str:
    """Resolve runs base: explicit arg > VLM_RL_RUN_DIR > PROJECT_DIR/runs > project runs/."""
    if run_dir_base is not None:
        return run_dir_base
    if base := os.environ.get("VLM_RL_RUN_DIR"):
        return base
    if project_dir := os.environ.get("PROJECT_DIR"):
        return os.path.join(project_dir, "runs")
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(_project_root, "runs")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    prompt_slug = sanitize_prompt_for_filename(args.vlm_goal)
    run_name = f"{args.env_id}__{args.exp_name}__{prompt_slug}__{args.seed}__{int(time.time())}"
    run_dir_base = _get_run_dir_base(args.run_dir_base)
    run_dir = os.path.join(run_dir_base, run_name)
    os.makedirs(run_dir, exist_ok=True)
    if args.track:
        import wandb

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
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    reward_model = RewardModel(envs.single_observation_space.shape).to(device)
    rm_optimizer = optim.Adam(reward_model.parameters(), lr=args.rm_lr)
    rating_dataset = ERLDataset()
    rm_criterion = nn.L1Loss()

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    prev_rollout_observations: list[np.ndarray] = []

    # Warm-start: gather a larger annotated dataset before PPO updates.
    initial_annotations = 0
    initial_mean_rating = 0.0
    if args.initial_annotation_target > 0:
        print(f"Starting initial annotation warmup for {args.initial_annotation_target} samples...")
        collected_ratings = []
        while initial_annotations < args.initial_annotation_target:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(next_obs)

            next_obs_np, _, terminations, truncations, _ = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)

            remaining = args.initial_annotation_target - initial_annotations
            obs_batch = next_obs_np[:remaining]
            ratings = annotate_observation_batch([obs for obs in obs_batch], args)
            for sampled, rating in zip(obs_batch, ratings):
                rating_dataset.add(sampled, rating)
                collected_ratings.append(rating)
            initial_annotations += len(obs_batch)

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            if initial_annotations % max(64, args.rating_batch_size) == 0:
                print(f"Warmup annotations collected: {initial_annotations}/{args.initial_annotation_target}")

        initial_mean_rating = float(np.mean(collected_ratings)) if collected_ratings else 0.0
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
            f"dataset={len(rating_dataset)} mean_rating={initial_mean_rating:.2f} "
            f"pretrain_loss={pretrain_loss:.4f}"
        )
        writer.add_scalar("erl/initial_annotations", initial_annotations, 0)
        writer.add_scalar("erl/initial_mean_vlm_rating", initial_mean_rating, 0)
        writer.add_scalar("erl/initial_pretrain_loss", pretrain_loss, 0)

    for iteration in range(1, args.num_iterations + 1):
        iteration_start_time = time.time()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Phase A: Data Collection & VLM Annotation (from previous rollout frames)
        annotations_added = 0
        mean_vlm_rating = 0.0
        if iteration % args.vlm_query_interval == 0 and len(prev_rollout_observations) > 0:
            num_to_rate = min(args.rating_batch_size, len(prev_rollout_observations))
            sampled_obs = random.sample(prev_rollout_observations, num_to_rate)
            sampled_ratings = annotate_observation_batch(sampled_obs, args)
            for sampled, rating in zip(sampled_obs, sampled_ratings):
                rating_dataset.add(sampled, rating)
            annotations_added = len(sampled_ratings)
            mean_vlm_rating = float(np.mean(sampled_ratings)) if sampled_ratings else 0.0

        # Phase B: Train Reward Model on stratified ratings
        rm_loss_value = 0.0
        if len(rating_dataset) > 0:
            rm_loss_value = train_reward_model(
                reward_model=reward_model,
                rm_optimizer=rm_optimizer,
                rating_dataset=rating_dataset,
                rm_criterion=rm_criterion,
                device=device,
                epochs=args.rm_train_epochs,
                batch_size=args.rating_batch_size,
            )

        # Phase C: PPO Rollout using RM-predicted rewards
        reward_model.eval()
        current_rollout_observations: list[np.ndarray] = []
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_np, _, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            current_rollout_observations.extend(next_obs_np.copy())

            with torch.no_grad():
                rm_rewards = reward_model(torch.tensor(next_obs_np, dtype=torch.float32, device=device))
            rewards[step] = rm_rewards.view(-1)

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            elif "episode" in infos:
                for i, done in enumerate(infos.get("_episode", [])):
                    if done:
                        ep_return = infos["episode"]["r"][i].item()
                        ep_length = infos["episode"]["l"][i].item()
                        print(f"global_step={global_step}, episodic_return={ep_return:.3f}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)

        prev_rollout_observations = current_rollout_observations
        rm_reward_mean = rewards.mean().item()
        rm_reward_std = rewards.std().item()

        # Phase D: PPO Update (unchanged core logic)
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
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

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        writer.add_scalar("erl/rm_loss", rm_loss_value, global_step)
        writer.add_scalar("erl/dataset_size", len(rating_dataset), global_step)
        writer.add_scalar("erl/annotations_added", annotations_added, global_step)
        writer.add_scalar("erl/mean_vlm_rating", mean_vlm_rating, global_step)
        writer.add_scalar("erl/rm_reward_mean", rm_reward_mean, global_step)
        writer.add_scalar("erl/rm_reward_std", rm_reward_std, global_step)
        for rating in range(1, 6):
            writer.add_scalar(f"erl/rating_bucket_{rating}", len(rating_dataset.buckets[rating]), global_step)

        iteration_s = time.time() - iteration_start_time
        print(
            f"[iter {iteration:04d}/{args.num_iterations}] "
            f"dataset={len(rating_dataset)} (+{annotations_added}, mean_rating={mean_vlm_rating:.2f}) "
            f"rm_loss={rm_loss_value:.4f} rm_reward_mean={rm_reward_mean:.4f} "
            f"rm_reward_std={rm_reward_std:.4f} iter_s={iteration_s:.2f}"
        )

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
