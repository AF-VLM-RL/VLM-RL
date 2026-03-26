import re
from collections import deque
from typing import List, Literal, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
)

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

_NUMBER_WORD_TO_INT = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _parse_vlm_rating(text: str) -> int:
    text_l = text.lower()
    tagged = re.search(r"(?:rating|score)\s*[:=]?\s*([1-5])\b", text_l)
    if tagged:
        return int(tagged.group(1))
    digit = re.search(r"\b([1-5])\b", text_l)
    if digit:
        return int(digit.group(1))
    for word, val in _NUMBER_WORD_TO_INT.items():
        if re.search(rf"\b{word}\b", text_l):
            return val
    return 3


def _extract_vlm_raw_score(text: str) -> float:
    text_l = text.lower()
    tagged = re.search(r"(?:rating|score)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text_l)
    if tagged:
        v = float(tagged.group(1))
        if 1 <= v <= 5:
            return v
        if 0 <= v <= 1:
            return 1 + 4 * v
        if 0 <= v <= 100:
            return 1 + 4 * (v / 100)
    digit = re.search(r"(-?\d+(?:\.\d+)?)", text_l)
    if digit:
        v = float(digit.group(1))
        if 1 <= v <= 5:
            return v
        if 0 <= v <= 1:
            return 1 + 4 * v
        if 0 <= v <= 100:
            return 1 + 4 * (v / 100)
    return float(_parse_vlm_rating(text))



# ---------------------------------------------------------------------------
# Shared VLM state (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_VLM_FS_PROCESSOR: Optional[AutoProcessor] = None
_VLM_FS_MODEL: Optional[AutoModelForVision2Seq] = None
_VLM_FS_DEVICE: Optional[torch.device] = None
_VLM_FS_MODEL_NAME: Optional[str] = None


def _init_vlm_frame_sequence(model_name: str, device: torch.device) -> None:
    global _VLM_FS_PROCESSOR, _VLM_FS_MODEL, _VLM_FS_DEVICE, _VLM_FS_MODEL_NAME
    if _VLM_FS_MODEL_NAME == model_name and _VLM_FS_MODEL is not None:
        return
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    _VLM_FS_PROCESSOR = AutoProcessor.from_pretrained(model_name)

    # Left-padding + valid pad token must be set before any batched call.
    tok = _VLM_FS_PROCESSOR.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    if device.type == "cuda":
        _VLM_FS_MODEL = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="cuda",
        )
    else:
        _VLM_FS_MODEL = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
    _VLM_FS_MODEL.eval()
    _VLM_FS_DEVICE = device
    _VLM_FS_MODEL_NAME = model_name


# ---------------------------------------------------------------------------
# Shared batched inference path — the ONLY place generate() is called
# ---------------------------------------------------------------------------

def _vlm_batch_score(
    images: List[Image.Image],
    cached_text: str,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    device: torch.device,
    max_new_tokens: int,
    reward_scale: float,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_samples: int,
) -> tuple[np.ndarray, List[str]]:
    """Score a batch of composite images with a single generate() call and return raw texts.

    Args:
        images: One composite PIL per batch row. Each composite encodes a **single** timeline:
            multiple RGB frames tiled (strip or grid) in **strict chronological order**
            (oldest→newest). Rows are independent envs/steps — order across rows is not
            a shared motion sequence.
        cached_text: Fully-templated prompt string built once at wrapper init.
            Replicated across the batch — all environments share the same goal.

    Returns:
        Float32 array of shape (len(images),) with rewards in [0, reward_scale].

    This function is the single generate() call-site.  Both
    BatchedVLMRewardVecWrapper (batch = num_envs) and
    VLMFrameSequenceRewardWrapper (batch = 1) go through here, so the inference
    path is identical regardless of how many environments are running.
    """
    n = len(images)
    texts = [cached_text] * n  # same prompt for every env

    # images=[img_0, img_1, ..., img_N] — the processor handles resizing,
    # pixel value normalisation, and building the attention mask in one shot.
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    num_samples = max(1, int(num_samples))
    rewards_accum = np.zeros(n, dtype=np.float32)
    all_texts: List[List[str]] = [[] for _ in range(n)]

    for _ in range(num_samples):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # With left-padding, generated tokens always occupy the last max_new_tokens
        # columns — correct for every row regardless of individual prompt length.
        generated_texts = processor.batch_decode(
            output_ids[:, -max_new_tokens:], skip_special_tokens=True
        )

        for i, text in enumerate(generated_texts):
            all_texts[i].append(text)
            raw = float(np.clip(_extract_vlm_raw_score(text.strip()), 1.0, 5.0))
            rewards_accum[i] += reward_scale * (raw - 1.0) / 4.0

    rewards = rewards_accum / float(num_samples)
    # Join multiple sampled texts for logging/debugging.
    joined_texts = [" || ".join([t.strip() for t in ts]) for ts in all_texts]
    return rewards, joined_texts


def score_rollout_composite_images(
    images: List[Image.Image],
    goal: str,
    layout: str,
    model_name: str,
    device: torch.device,
    reward_scale: float,
    max_new_tokens: int,
    vlm_do_sample: bool,
    vlm_temperature: float,
    vlm_top_p: float,
    vlm_num_samples: int,
    chunk_size: int,
) -> np.ndarray:
    """Score many trajectory-step composites with chunked batched VLM calls.

    Used when rollout collects one composite per (step, env) and scoring is
    deferred until after the rollout loop (fewer, larger batches vs. per-step).

    Args:
        images: Flat list in row-major order (step 0 all envs, step 1 all envs, ...).
        chunk_size: Max images per single ``generate()`` (tune for GPU memory).

    Returns:
        Float32 array of shape ``(len(images),)`` in the same order as ``images``.
    """
    if not images:
        return np.zeros(0, dtype=np.float32)

    _init_vlm_frame_sequence(model_name, device)
    processor = _VLM_FS_PROCESSOR
    model = _VLM_FS_MODEL
    dev = _VLM_FS_DEVICE
    assert processor is not None and model is not None and dev is not None

    prompt_text = _build_sequence_rating_prompt(goal, layout=layout)
    single_msg = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
    ]
    cached_text = processor.apply_chat_template(
        single_msg, tokenize=False, add_generation_prompt=True
    )

    chunk_size = max(1, int(chunk_size))
    parts: List[np.ndarray] = []
    for start in range(0, len(images), chunk_size):
        chunk = images[start : start + chunk_size]
        r, _ = _vlm_batch_score(
            images=chunk,
            cached_text=cached_text,
            processor=processor,
            model=model,
            device=dev,
            max_new_tokens=max_new_tokens,
            reward_scale=reward_scale,
            do_sample=bool(vlm_do_sample),
            temperature=float(vlm_temperature),
            top_p=float(vlm_top_p),
            num_samples=int(vlm_num_samples),
        )
        parts.append(r)
    return np.concatenate(parts, axis=0)


def _vlm_video_batch_score(
    frame_sequences: List[List[np.ndarray]],
    prompt_text: str,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    device: torch.device,
    max_new_tokens: int,
    reward_scale: float,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_samples: int,
    video_fps: float,
) -> tuple[np.ndarray, List[str]]:
    if process_vision_info is None:
        raise ImportError('qwen_vl_utils is required for video input format. Install qwen-vl-utils.')

    n = len(frame_sequences)
    messages = []
    texts = []
    for seq in frame_sequences:
        pil_frames = [Image.fromarray(np.asarray(f).astype(np.uint8)).convert('RGB') for f in seq]
        msg = [{
            'role': 'user',
            'content': [
                {'type': 'video', 'video': pil_frames, 'fps': float(video_fps)},
                {'type': 'text', 'text': prompt_text},
            ],
        }]
        messages.append(msg)
        texts.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    num_samples = max(1, int(num_samples))
    rewards_accum = np.zeros(n, dtype=np.float32)
    all_texts: List[List[str]] = [[] for _ in range(n)]

    for _ in range(num_samples):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        generated_texts = processor.batch_decode(output_ids[:, -max_new_tokens:], skip_special_tokens=True)
        for i, text in enumerate(generated_texts):
            all_texts[i].append(text)
            raw = float(np.clip(_extract_vlm_raw_score(text.strip()), 1.0, 5.0))
            rewards_accum[i] += reward_scale * (raw - 1.0) / 4.0

    rewards = rewards_accum / float(num_samples)
    joined_texts = [' || '.join([t.strip() for t in ts]) for ts in all_texts]
    return rewards, joined_texts


def score_rollout_frame_sequences_video(
    frame_sequences: List[List[np.ndarray]],
    goal: str,
    layout: str,
    model_name: str,
    device: torch.device,
    reward_scale: float,
    max_new_tokens: int,
    vlm_do_sample: bool,
    vlm_temperature: float,
    vlm_top_p: float,
    vlm_num_samples: int,
    chunk_size: int,
    video_fps: float = 8.0,
) -> np.ndarray:
    if not frame_sequences:
        return np.zeros(0, dtype=np.float32)

    _init_vlm_frame_sequence(model_name, device)
    processor = _VLM_FS_PROCESSOR
    model = _VLM_FS_MODEL
    dev = _VLM_FS_DEVICE
    assert processor is not None and model is not None and dev is not None

    prompt_text = _build_sequence_rating_prompt(goal, layout=layout)
    chunk_size = max(1, int(chunk_size))
    parts: List[np.ndarray] = []
    for start in range(0, len(frame_sequences), chunk_size):
        chunk = frame_sequences[start : start + chunk_size]
        r, _ = _vlm_video_batch_score(
            frame_sequences=chunk,
            prompt_text=prompt_text,
            processor=processor,
            model=model,
            device=dev,
            max_new_tokens=max_new_tokens,
            reward_scale=reward_scale,
            do_sample=bool(vlm_do_sample),
            temperature=float(vlm_temperature),
            top_p=float(vlm_top_p),
            num_samples=int(vlm_num_samples),
            video_fps=float(video_fps),
        )
        parts.append(r)
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# CLIP batched rollout scoring (contrastive, no generation)
# ---------------------------------------------------------------------------

_CLIP_ROLLOUT_MODEL: Optional[CLIPModel] = None
_CLIP_ROLLOUT_PROCESSOR: Optional[CLIPProcessor] = None
_CLIP_ROLLOUT_DEVICE: Optional[torch.device] = None
_CLIP_ROLLOUT_MODEL_ID: Optional[str] = None


def _init_clip_rollout_scorer(model_id: str, device: torch.device) -> None:
    global _CLIP_ROLLOUT_MODEL, _CLIP_ROLLOUT_PROCESSOR, _CLIP_ROLLOUT_DEVICE, _CLIP_ROLLOUT_MODEL_ID
    if _CLIP_ROLLOUT_MODEL_ID == model_id and _CLIP_ROLLOUT_MODEL is not None:
        return
    _CLIP_ROLLOUT_PROCESSOR = CLIPProcessor.from_pretrained(model_id)
    _CLIP_ROLLOUT_MODEL = CLIPModel.from_pretrained(model_id).to(device)
    if device.type == "cuda":
        _CLIP_ROLLOUT_MODEL = _CLIP_ROLLOUT_MODEL.half()
    _CLIP_ROLLOUT_MODEL.eval()
    _CLIP_ROLLOUT_DEVICE = device
    _CLIP_ROLLOUT_MODEL_ID = model_id
    print(f"CLIP rollout scorer: model={model_id}, device={device}, fp16={device.type == 'cuda'}")


def _clip_batch_score(
    images: List[Image.Image],
    goal: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    reward_scale: float,
    text_features_cache: Optional[torch.Tensor],
) -> tuple[np.ndarray, torch.Tensor]:
    """One forward pass: image batch × fixed text goal → cosine similarities → rewards.

    Each row is one composite PIL (e.g. tiled frames). Rewards are
    ``reward_scale * (cosine_sim + 1) / 2`` in ``[0, reward_scale]``.

    Returns:
        rewards: shape (len(images),).
        text_features_cache: normalized text features [1, dim] for reuse (same goal).
    """
    n = len(images)
    if n == 0:
        return np.zeros(0, dtype=np.float32), text_features_cache

    use_fp16 = device.type == "cuda"
    with torch.no_grad():
        if text_features_cache is None:
            t_inputs = processor(text=[goal], return_tensors="pt", padding=True)
            t_inputs = {k: v.to(device) for k, v in t_inputs.items()}
            tf = model.get_text_features(**t_inputs)
            tf = F.normalize(tf.float(), dim=-1)
            if use_fp16:
                tf = tf.half()
            text_features_cache = tf
        else:
            tf = text_features_cache

        img_inputs = processor(images=images, return_tensors="pt")
        img_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in img_inputs.items()}
        if use_fp16 and "pixel_values" in img_inputs:
            img_inputs["pixel_values"] = img_inputs["pixel_values"].half()

        imf = model.get_image_features(**img_inputs)
        imf = F.normalize(imf.float(), dim=-1)
        sim = (imf @ tf.float().T).squeeze(-1)
        rewards_t = reward_scale * (sim + 1.0) / 2.0
        rewards = rewards_t.detach().float().cpu().numpy().astype(np.float32)

    return rewards, text_features_cache


def _clip_batch_score_multi_goal(
    images: List[Image.Image],
    goals: List[str],
    goal_weights: List[float],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    reward_scale: float,
    text_features_cache: Optional[torch.Tensor],
) -> tuple[np.ndarray, Optional[torch.Tensor]]:
    """Score images against multiple goals; return weighted sum of (cos_sim+1)/2 per goal.

    goals: list of prompt strings.
    goal_weights: weights for each goal (will be normalized to sum to 1).
    text_features_cache: [num_goals, dim] normalized text features, or None.
    """
    n = len(images)
    n_goals = len(goals)
    if n == 0 or n_goals == 0:
        return np.zeros(max(0, n), dtype=np.float32), text_features_cache

    weights = np.asarray(goal_weights, dtype=np.float32)
    if len(weights) != n_goals:
        weights = np.ones(n_goals, dtype=np.float32) / n_goals
    weights = weights / weights.sum()
    weights_t = torch.from_numpy(weights).to(device)

    use_fp16 = device.type == "cuda"
    with torch.no_grad():
        if text_features_cache is None:
            t_inputs = processor(text=goals, return_tensors="pt", padding=True)
            t_inputs = {k: v.to(device) for k, v in t_inputs.items()}
            tf = model.get_text_features(**t_inputs)
            tf = F.normalize(tf.float(), dim=-1)
            if use_fp16:
                tf = tf.half()
            text_features_cache = tf
        else:
            tf = text_features_cache

        img_inputs = processor(images=images, return_tensors="pt")
        img_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in img_inputs.items()}
        if use_fp16 and "pixel_values" in img_inputs:
            img_inputs["pixel_values"] = img_inputs["pixel_values"].half()

        imf = model.get_image_features(**img_inputs)
        imf = F.normalize(imf.float(), dim=-1)
        sim = imf @ tf.float().T
        scaled = (sim + 1.0) / 2.0
        rewards_t = (scaled * weights_t).sum(dim=-1) * reward_scale
        rewards = rewards_t.detach().float().cpu().numpy().astype(np.float32)

    return rewards, text_features_cache


def score_rollout_composite_images_clip(
    images: List[Image.Image],
    goal: str,
    clip_model_name: str,
    device: torch.device,
    reward_scale: float,
    chunk_size: int,
) -> np.ndarray:
    """Like ``score_rollout_composite_images`` but CLIP image–text similarity (batched, chunked).

    No generation: encodes the goal text once, then for each chunk runs a single
    ``get_image_features`` batch against that text embedding. ``layout`` is unused
    (composites are arbitrary RGB for CLIP).

    Args:
        images: Flat list of composite PILs (e.g. step-major × env-minor).
        chunk_size: Max images per CLIP forward (tune for GPU memory).

    Returns:
        Float32 array of shape ``(len(images),)``, same order as ``images``.
    """
    if not images:
        return np.zeros(0, dtype=np.float32)

    _init_clip_rollout_scorer(clip_model_name, device)
    model = _CLIP_ROLLOUT_MODEL
    processor = _CLIP_ROLLOUT_PROCESSOR
    dev = _CLIP_ROLLOUT_DEVICE
    assert model is not None and processor is not None and dev is not None

    chunk_size = max(1, int(chunk_size))
    parts: List[np.ndarray] = []
    text_features_cache: Optional[torch.Tensor] = None
    for start in range(0, len(images), chunk_size):
        chunk = images[start : start + chunk_size]
        r, text_features_cache = _clip_batch_score(
            images=chunk,
            goal=goal,
            processor=processor,
            model=model,
            device=dev,
            reward_scale=reward_scale,
            text_features_cache=text_features_cache,
        )
        parts.append(r)
    return np.concatenate(parts, axis=0)


def score_rollout_composite_images_clip_multi_goal(
    images: List[Image.Image],
    goals: List[str],
    goal_weights: List[float],
    clip_model_name: str,
    device: torch.device,
    reward_scale: float,
    chunk_size: int,
) -> np.ndarray:
    """CLIP scoring with multiple goals: reward = reward_scale * weighted_sum of normalized cos_sim.

    Args:
        goals: List of text prompts (e.g. ["upright posture", "smooth gait", "forward progress"]).
        goal_weights: Weights per goal; normalized to sum to 1.

    Returns:
        Float32 array of shape (len(images),).
    """
    if not images or not goals:
        return np.zeros(len(images) if images else 0, dtype=np.float32)

    _init_clip_rollout_scorer(clip_model_name, device)
    model = _CLIP_ROLLOUT_MODEL
    processor = _CLIP_ROLLOUT_PROCESSOR
    dev = _CLIP_ROLLOUT_DEVICE
    assert model is not None and processor is not None and dev is not None

    chunk_size = max(1, int(chunk_size))
    parts: List[np.ndarray] = []
    text_features_cache: Optional[torch.Tensor] = None
    for start in range(0, len(images), chunk_size):
        chunk = images[start : start + chunk_size]
        r, text_features_cache = _clip_batch_score_multi_goal(
            images=chunk,
            goals=goals,
            goal_weights=goal_weights,
            processor=processor,
            model=model,
            device=dev,
            reward_scale=reward_scale,
            text_features_cache=text_features_cache,
        )
        parts.append(r)
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _frame_to_pil(arr: np.ndarray, size: int = 224) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB").resize((size, size))


def _frames_to_grid(frames: List[np.ndarray], grid_size: int = 2) -> Image.Image:
    """Tile frames in row-major order: time increases left→right, then top→bottom."""
    if not frames:
        raise ValueError("frames must be non-empty")
    n = min(len(frames), grid_size * grid_size)
    imgs = [_frame_to_pil(f) for f in frames[:n]]
    rows = []
    for r in range(grid_size):
        row_imgs = imgs[r * grid_size : (r + 1) * grid_size]
        if len(row_imgs) < grid_size:
            row_imgs += [Image.new("RGB", (224, 224), (128, 128, 128))] * (
                grid_size - len(row_imgs)
            )
        rows.append(np.concatenate([np.array(img) for img in row_imgs], axis=1))
    return Image.fromarray(np.concatenate(rows, axis=0))


def _frames_to_strip(frames: List[np.ndarray], frame_size: int = 224) -> Image.Image:
    """Concatenate frames left→right; ``frames`` must be oldest→newest."""
    if not frames:
        raise ValueError("frames must be non-empty")
    imgs = [_frame_to_pil(f, size=frame_size) for f in frames]
    return Image.fromarray(np.concatenate([np.array(img) for img in imgs], axis=1))


def _build_sequence_rating_prompt(goal: str, layout: str = "strip") -> str:
    if layout == "strip":
        order_help = (
            "The image is ONE temporal sequence: panels are consecutive simulation frames, "
            "ordered left to right in time (left = earliest, right = latest). "
            "Use motion and changes from early to late frames when you judge progress and stability.\n"
        )
    elif layout == "grid":
        order_help = (
            "The image is ONE temporal sequence tiled in a grid: time increases in reading order — "
            "start at the top-left cell, go left to right along each row, then continue on the next row down "
            "(top-left = earliest, bottom-right of the filled cells = latest). "
            "Use how the pose and motion evolve across that order when you score.\n"
        )
    else:
        order_help = (
            "The image is ONE temporal sequence of consecutive simulation frames in a fixed spatial order; "
            "infer earliest→latest from that layout and use motion across time when you score.\n"
        )
    return (
        f"{order_help}Carefully look at these frames from a 2D physics simulation.\n"
        "The character is a simple robot with a small central torso and two jointed legs, but no arms.\n"
        f"Goal: {goal}\n"
        "IMPORTANT:\n"
        "- Do NOT assume the robot is walking correctly.\n"
        "- It might be standing still, moving backward, wobbling, or falling.\n"
        "- Falling is always BAD: if the torso is near the ground or the body is lying sideways, treat it as a failure.\n"
        "- Score MUST be based on what is visible in the frames.\n"
        "- Consider these factors when scoring:\n"
        "  - Uprightness: torso stays upright above the legs (falling = very low score)\n"
        "  - Forward progress: earliest→latest frames show clear motion to the right\n"
        "  - Stability: smooth gait vs jittering/standing still\n"
        "  - Leg movement: alternating stepping pattern (one leg swings forward/up while the other supports) rather than both legs flailing together\n"
        "First, describe what ACTUALLY happens in one short sentence.\n"
        "Then describe each factor in a few words:\n"
        "Uprightness: <e.g., upright / leaning / falling / on ground>\n"
        "ForwardProgress: <e.g., clear rightward progress / slight / none / backward>\n"
        "Stability: <e.g., smooth gait / jittery / stumbling / static>\n"
        "LegMovement: <e.g., alternating steps / shuffling / kicking / flailing / unclear>\n"
        "Then give a rating from 1 to 5 for how well the sequence matches the goal.\n"
        "Use the full range (1,2,3,4,5) and avoid always using 4 or 5.\n"
        "Output format:\n"
        "Description: <one short sentence>\n"
        "Uprightness: <few words>\n"
        "ForwardProgress: <few words>\n"
        "Stability: <few words>\n"
        "LegMovement: <few words>\n"
        "Rating: <1-5>"
    )


# ---------------------------------------------------------------------------
# FrameBufferWrapper — one per environment, no VLM calls
# ---------------------------------------------------------------------------

class FrameBufferWrapper(gym.Wrapper):
    """Buffers rendered frames and assembles composite images on demand.

    Contains no VLM logic.  BatchedVLMRewardVecWrapper calls get_composite_image()
    on all environments simultaneously, then scores them in one generate() call.
    """

    def __init__(
        self,
        env: gym.Env,
        max_frames: int = 4,
        layout: str = "strip",
        grid_size: int = 2,
        frame_stride: int = 1,
    ):
        super().__init__(env)
        self.max_frames = max_frames
        self.layout = layout
        self.grid_size = grid_size
        self.frame_stride = max(1, int(frame_stride))
        self.frame_buffer: deque = deque(maxlen=max_frames)
        self._since_last_append = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frame_buffer.clear()
        self._since_last_append = 0
        frame = self.env.render()
        if frame is not None:
            self.frame_buffer.append(np.asarray(frame).copy())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        if frame is not None:
            self._since_last_append += 1
            if self._since_last_append >= self.frame_stride:
                self.frame_buffer.append(np.asarray(frame).copy())
                self._since_last_append = 0

        if terminated or truncated:
            info["terminal_composite"] = self.get_composite_image()
            info["terminal_frames"] = self.get_frame_sequence()

        return obs, reward, terminated, truncated, info

    def get_frame_sequence(self) -> List[np.ndarray]:
        return [np.asarray(f).copy() for f in self.frame_buffer]

    def get_composite_image(self) -> Image.Image:
        """Return recent RGB frames as one PIL image (oldest→newest in buffer order).

        ``deque`` iteration is left-to-right oldest→newest. When ``frame_stride > 1``,
        each stored panel is every ``frame_stride``-th env step.
        """
        frames = list(self.frame_buffer)
        if not frames:
            return Image.new("RGB", (224, 224), (0, 0, 0))
        if self.layout == "strip":
            return _frames_to_strip(frames)
        return _frames_to_grid(frames, grid_size=self.grid_size)


# ---------------------------------------------------------------------------
# BatchedVLMRewardVecWrapper — scores all envs in one generate() call
# ---------------------------------------------------------------------------

class BatchedVLMRewardVecWrapper(gym.Wrapper):
    """Replaces environment rewards with VLM scores across all parallel envs.

    On every environment step:
      1. Calls env.call("get_composite_image") to collect one PIL image from
         every FrameBufferWrapper instance — one image per environment.
      2. Passes images=[img_0, img_1, ..., img_N] into _vlm_batch_score, which
         issues a single batched generate() call covering all N environments.
      3. Distributes the resulting per-environment scores as rewards.

    N environments → 1 VLM forward pass, not N passes.
    """

    def __init__(
        self,
        envs: gym.vector.VectorEnv,
        goal: str,
        device: torch.device,
        model_name: str,
        skip_frames: int,
        reward_scale: float,
        layout: str,
        max_new_tokens: int = 12,
        vlm_do_sample: bool = True,
        vlm_temperature: float = 0.7,
        vlm_top_p: float = 0.9,
        vlm_num_samples: int = 3,
    ):
        super().__init__(envs)
        # NOTE: skip_frames is kept in the signature for backwards compatibility only.
        self.reward_scale = reward_scale
        self.max_new_tokens = max_new_tokens
        self.vlm_do_sample = bool(vlm_do_sample)
        self.vlm_temperature = float(vlm_temperature)
        self.vlm_top_p = float(vlm_top_p)
        self.vlm_num_samples = int(vlm_num_samples)
        self.num_envs = envs.num_envs
        self.last_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.last_texts: List[str] = [""] * self.num_envs

        _init_vlm_frame_sequence(model_name, device)
        self._processor = _VLM_FS_PROCESSOR
        self._model = _VLM_FS_MODEL
        self._device = _VLM_FS_DEVICE

        # Build and cache the templated prompt once — shared across all envs.
        prompt_text = _build_sequence_rating_prompt(goal, layout=layout)
        single_msg = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
        ]
        self._cached_text = self._processor.apply_chat_template(
            single_msg, tokenize=False, add_generation_prompt=True
        )

        print(
            f"BatchedVLMRewardVecWrapper: num_envs={self.num_envs}, "
            f"skip={skip_frames}, max_new_tokens={max_new_tokens}, model={model_name}"
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_rewards = self._compute_batched_rewards()
        return obs, info

    def step(self, action):
        obs, _, terminations, truncations, infos = self.env.step(action)
        # Recompute VLM rewards on *every* step for all environments.
        self.last_rewards = self._compute_batched_rewards()

        rewards = self.last_rewards.copy()
        if isinstance(infos, dict):
            infos["vlm_reward"] = self.last_rewards.copy()

        return obs, rewards, terminations, truncations, infos

    def _compute_batched_rewards(self) -> np.ndarray:
        # Collect one composite image from every parallel FrameBufferWrapper.
        # gym.vector.VectorEnv.call may return a tuple; convert to a plain list
        # so the transformers image processor receives a Python list, not tuple.
        raw_images = self.env.call("get_composite_image")
        composite_images: List[Image.Image] = list(raw_images)

        # Single generate() call — batch size = num_envs.
        rewards, texts = _vlm_batch_score(
            images=composite_images,
            cached_text=self._cached_text,
            processor=self._processor,
            model=self._model,
            device=self._device,
            max_new_tokens=self.max_new_tokens,
            reward_scale=self.reward_scale,
            do_sample=self.vlm_do_sample,
            temperature=self.vlm_temperature,
            top_p=self.vlm_top_p,
            num_samples=self.vlm_num_samples,
        )
        self.last_texts = list(texts)
        return rewards


# ---------------------------------------------------------------------------
# VLMFrameSequenceRewardWrapper — single-env wrapper, same inference path
# ---------------------------------------------------------------------------

class VLMFrameSequenceRewardWrapper(gym.Wrapper):
    """Single-environment wrapper that scores frame sequences via a VLM.

    Uses _vlm_batch_score with batch size = 1, so the inference path is
    identical to BatchedVLMRewardVecWrapper.  For multi-environment training
    prefer FrameBufferWrapper + BatchedVLMRewardVecWrapper.
    """

    def __init__(
        self,
        env: gym.Env,
        goal: str,
        device: torch.device,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        skip_frames: int = 8,
        max_frames: int = 4,
        layout: Literal["strip", "grid"] = "strip",
        grid_size: int = 2,
        reward_scale: float = 1.0,
        max_new_tokens: int = 12,
        vlm_do_sample: bool = True,
        vlm_temperature: float = 0.7,
        vlm_top_p: float = 0.9,
        vlm_num_samples: int = 3,
    ):
        super().__init__(env)
        # NOTE: skip_frames is kept for backwards compatibility but is unused.
        self.skip_frames = skip_frames
        self.max_frames = max_frames
        self.layout = layout
        self.grid_size = grid_size
        self.reward_scale = reward_scale
        self.max_new_tokens = max_new_tokens
        self.vlm_do_sample = bool(vlm_do_sample)
        self.vlm_temperature = float(vlm_temperature)
        self.vlm_top_p = float(vlm_top_p)
        self.vlm_num_samples = int(vlm_num_samples)
        self.frame_buffer: deque = deque(maxlen=max_frames)
        self.last_reward = 0.0

        _init_vlm_frame_sequence(model_name, device)
        self._processor = _VLM_FS_PROCESSOR
        self._model = _VLM_FS_MODEL
        self._device = _VLM_FS_DEVICE

        prompt_text = _build_sequence_rating_prompt(goal, layout=layout)
        single_msg = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
        ]
        self._cached_text = self._processor.apply_chat_template(
            single_msg, tokenize=False, add_generation_prompt=True
        )

        print(
            f"VLMFrameSequenceRewardWrapper: goal='{goal}', skip={skip_frames}, "
            f"max_frames={max_frames}, layout={layout}, model={model_name}"
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frame_buffer.clear()
        frame = self.env.render()
        assert frame is not None, "env.render() returned None. Use render_mode='rgb_array'."
        self.frame_buffer.append(np.asarray(frame).copy())
        self.last_reward = self._compute_reward()
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        assert frame is not None, "env.render() returned None during step."
        self.frame_buffer.append(np.asarray(frame).copy())
        # Recompute VLM reward on every step.
        self.last_reward = self._compute_reward()
        info["vlm_reward"] = self.last_reward
        return obs, self.last_reward, terminated, truncated, info

    def _compute_reward(self) -> float:
        frames = list(self.frame_buffer)
        if not frames:
            return 0.0
        composite = (
            _frames_to_strip(frames)
            if self.layout == "strip"
            else _frames_to_grid(frames, grid_size=self.grid_size)
        )
        # Delegate to the shared batched path — batch size = 1.
        rewards, _texts = _vlm_batch_score(
            images=[composite],
            cached_text=self._cached_text,
            processor=self._processor,
            model=self._model,
            device=self._device,
            max_new_tokens=self.max_new_tokens,
            reward_scale=self.reward_scale,
            do_sample=self.vlm_do_sample,
            temperature=self.vlm_temperature,
            top_p=self.vlm_top_p,
            num_samples=self.vlm_num_samples,
        )
        return float(rewards[0])


# ---------------------------------------------------------------------------
# VLMRewardWrapper — CLIP-based single-frame reward (unchanged)
# ---------------------------------------------------------------------------

class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, text_goal, device, skip_frames=8):
        super().__init__(env)
        self.device = device
        # NOTE: skip_frames is kept for backwards compatibility but no longer
        # controls when the VLM is queried; rewards are computed every step.
        self.skip_frames = skip_frames
        self.last_reward = 0.0

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).half()
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        with torch.no_grad():
            text_inputs = tokenizer([text_goal], padding=True, return_tensors="pt")
            text_feats = self.model.get_text_features(**text_inputs.to(device))
            self.text_features = (text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)).half()

        self.mean = torch.tensor([0.4814, 0.4578, 0.4082], device=device, dtype=torch.float16).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.2686, 0.2613, 0.2757], device=device, dtype=torch.float16).view(1, 3, 1, 1)

        print(f"VLMRewardWrapper (CLIP): FP16=True, skip={skip_frames}, goal='{text_goal}'")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self.env.render()
        assert frame is not None, "env.render() returned None. Use render_mode='rgb_array'."
        self.last_reward = self._compute_reward(frame)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        # Recompute VLM reward on every step.
        frame = self.env.render()
        assert frame is not None, "env.render() returned None during step."
        self.last_reward = self._compute_reward(frame)
        info["vlm_reward"] = self.last_reward
        return obs, self.last_reward, terminated, truncated, info

    def _compute_reward(self, frame_array: np.ndarray) -> float:
        img = torch.tensor(frame_array, device=self.device, dtype=torch.float16)
        img = img.permute(2, 0, 1).unsqueeze(0) / 255.0
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        img = (img - self.mean) / self.std
        with torch.no_grad():
            feats = self.model.get_image_features(img)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            return (feats @ self.text_features.T).item()