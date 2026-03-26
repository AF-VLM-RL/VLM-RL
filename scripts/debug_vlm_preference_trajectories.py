import os
import sys
import time
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.wrappers import FrameBufferWrapper  # noqa: E402


def _save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def collect_composites(
    env_id: str,
    num_trajs: int,
    traj_length: int,
    out_dir: str,
    max_frames: int = 16,
    layout: str = "grid",
    grid_size: int = 4,
) -> List[Image.Image]:
    """Collect composite images from random Walker2d trajectories."""
    env = FrameBufferWrapper(
        gym.wrappers.FlattenObservation(
            gym.make(env_id, render_mode="rgb_array")
        ),
        max_frames=max_frames,
        layout=layout,
        grid_size=grid_size,
    )
    obs, _ = env.reset(seed=1)

    composites: List[Image.Image] = []
    for ti in range(num_trajs):
        frames: List[Image.Image] = []
        for step in range(traj_length):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            comp = env.get_composite_image()
            frames.append(comp)
            if done:
                obs, _ = env.reset()
                break

        # Save final composite for this trajectory
        if frames:
            final_comp = frames[-1]
            composites.append(final_comp)
            _save_image(
                final_comp, os.path.join(out_dir, f"traj_{ti:03d}_final.png")
            )

    env.close()
    return composites


def query_vlm_on_pairs(
    composites: List[Image.Image],
    model_name: str,
    device: str,
    goal: str,
    out_dir: str,
    max_new_tokens: int = 64,
):
    """Form pairs of composites, run two-stage prompting, and save analyses + labels."""
    from transformers import AutoModelForVision2Seq, AutoProcessor  # lazy

    dev = torch.device(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map="cuda" if dev.type == "cuda" else None,
    )
    model.eval()

    def build_prompt() -> str:
        return (
            "You will see two short video strips from a 2D physics simulation, labelled A and B.\n"
            "Each strip shows a simple robot with a small central torso and two jointed legs but no arms.\n"
            f"Goal: {goal}\n"
            "Decide which strip better matches the goal, based ONLY on what you see.\n"
            "Consider uprightness (staying above the legs), forward progress to the right, stability of gait, "
            "and clear alternating leg movement (one leg stepping forward while the other supports).\n"
            "First, describe the key differences between A and B in 1-2 sentences.\n"
        )

    n = len(composites)
    if n < 2:
        print("Need at least 2 composites to form pairs.")
        return

    os.makedirs(out_dir, exist_ok=True)
    prompt = build_prompt()

    pair_log_path = os.path.join(out_dir, "vlm_pair_responses.txt")
    with open(pair_log_path, "w") as f:
        f.write("# idx_a\tidx_b\tanalysis\tlabel\n")

        # Simple pairing: (0,1), (2,3), ...
        num_pairs = n // 2
        for i in range(num_pairs):
            ia = 2 * i
            ib = 2 * i + 1
            images = [composites[ia], composites[ib]]

            # Stage 1: analysis
            msg1 = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text1 = processor.apply_chat_template(
                msg1, tokenize=False, add_generation_prompt=True
            )
            inputs1 = processor(
                text=[text1],
                images=[images],
                padding=True,
                return_tensors="pt",
            )
            inputs1 = {
                k: v.to(dev) if torch.is_tensor(v) else v
                for k, v in inputs1.items()
            }

            with torch.no_grad():
                out_ids1 = model.generate(
                    **inputs1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            in_len1 = inputs1["input_ids"].shape[1]
            analysis = processor.batch_decode(
                out_ids1[:, in_len1:], skip_special_tokens=True
            )[0].strip()

            # Stage 2: classification based on analysis
            cls_prompt = (
                "You are given an analysis comparing two trajectories A and B:\n"
                f"{analysis}\n\n"
                "Based on this analysis, output a single integer label:\n"
                "1  if A is better than B for the goal.\n"
                "0  if B is better than A for the goal.\n"
                "-1 if you cannot decide or they are equally good/bad.\n"
                "Output only the integer, nothing else."
            )
            tok_inputs = processor.tokenizer(
                [cls_prompt], return_tensors="pt", padding=True
            )
            tok_inputs = {
                k: v.to(dev) if torch.is_tensor(v) else v
                for k, v in tok_inputs.items()
            }
            with torch.no_grad():
                out_ids2 = model.generate(
                    **tok_inputs,
                    do_sample=False,
                    max_new_tokens=8,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            in_len2 = tok_inputs["input_ids"].shape[1]
            label_text = processor.tokenizer.batch_decode(
                out_ids2[:, in_len2:], skip_special_tokens=True
            )[0].strip()

            f.write(f"{ia}\t{ib}\t{analysis}\t{label_text}\n")
            print(f"Pair ({ia},{ib}) -> analysis: {analysis} | label: {label_text}")


def main():
    out_root = "vlm_pair_debug"
    timestamp = int(time.time())
    out_dir = os.path.join(out_root, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    env_id = "Walker2d-v4"
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    goal = (
        "a 2D robot with a small central torso and two jointed legs but no arms, "
        "staying upright and walking steadily to the right over time without falling"
    )

    print(f"Saving composites to {out_dir}")
    composites = collect_composites(
        env_id=env_id,
        num_trajs=16,
        traj_length=64,
        out_dir=out_dir,
        max_frames=16,
        layout="grid",
        grid_size=4,
    )

    print("Querying VLM on trajectory pairs...")
    query_vlm_on_pairs(
        composites=composites,
        model_name=model_name,
        device=device,
        goal=goal,
        out_dir=out_dir,
        max_new_tokens=64,
    )
    print("Done.")


if __name__ == "__main__":
    main()

