import argparse
import json
import os
import random
import re
from collections import Counter
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForVision2Seq, AutoProcessor


NUMBER_WORD_TO_INT = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def parse_rating(text: str) -> int:
    text_l = text.lower()
    # Prefer an explicit "rating: X" pattern when available.
    tagged_match = re.search(r"(?:rating|score)\s*[:=]?\s*([1-5])\b", text_l)
    if tagged_match:
        return int(tagged_match.group(1))

    digit_match = re.search(r"\b([1-5])\b", text_l)
    if digit_match:
        return int(digit_match.group(1))

    for word, value in NUMBER_WORD_TO_INT.items():
        if re.search(rf"\b{word}\b", text_l):
            return value

    return 3


def extract_raw_score(text: str) -> float:
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
    return float(parse_rating(text))


def build_rating_prompt(goal: str, rubric_variant: int = 0) -> str:
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


def extract_description(text: str) -> str:
    for line in text.splitlines():
        if ":" in line and line.lower().startswith("description"):
            return line.split(":", 1)[1].strip()
    return text.strip()


def capture_env_frame(env_id: str, seed: int, random_steps: int) -> np.ndarray:
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    _ = obs
    for _ in range(random_steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    frame = env.render()
    env.close()
    if frame is None:
        raise RuntimeError("env.render() returned None. Ensure render_mode='rgb_array'.")
    return frame


def load_qwen_model(model_name: str, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    return processor, model


def query_qwen_rating(
    image: Image.Image,
    goal: str,
    processor,
    model,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    rubric_variant: int,
):
    prompt = build_rating_prompt(goal, rubric_variant=rubric_variant)

    if hasattr(processor, "apply_chat_template"):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    else:
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=max(1e-5, temperature) if do_sample else None,
            top_p=top_p if do_sample else None,
            max_new_tokens=max_new_tokens,
        )

    if "input_ids" in inputs:
        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = output_ids[:, prompt_len:]
        generated_text = processor.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()
    else:
        generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    raw_score = float(np.clip(extract_raw_score(generated_text), 1.0, 5.0))
    rating = int(np.clip(parse_rating(generated_text), 1, 5))
    return rating, raw_score, generated_text


def save_annotated_image(image: Image.Image, rating: int, raw_text: str, output_path: str):
    img = image.copy().convert("RGB")
    drawer = ImageDraw.Draw(img)
    desc = extract_description(raw_text)
    if len(desc) > 80:
        desc = desc[:77] + "..."
    label = f"Qwen rating: {rating}/5 | {desc}"
    drawer.rectangle([(0, 0), (img.width, 24)], fill=(0, 0, 0))
    drawer.text((6, 6), label, fill=(255, 255, 255))
    img.save(output_path)


def annotate_image_batch(images, goal, processor, model, device, args):
    discrete = []
    raw_scores = []
    raw_texts = []
    for image in images:
        rating, raw_score, raw_text = query_qwen_rating(
            image=image,
            goal=goal,
            processor=processor,
            model=model,
            device=device,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.vlm_do_sample,
            temperature=args.vlm_temperature,
            top_p=args.vlm_top_p,
            rubric_variant=0,
        )
        discrete.append(rating)
        raw_scores.append(raw_score)
        raw_texts.append(raw_text)

    if len(discrete) == 0:
        return [], [], []

    dominant_fraction = Counter(discrete).most_common(1)[0][1] / len(discrete)
    if dominant_fraction >= args.vlm_collapse_threshold and len(discrete) > 1:
        discrete = []
        raw_scores = []
        raw_texts = []
        for image in images:
            rating, raw_score, raw_text = query_qwen_rating(
                image=image,
                goal=goal,
                processor=processor,
                model=model,
                device=device,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=max(0.3, args.vlm_temperature),
                top_p=args.vlm_top_p,
                rubric_variant=1,
            )
            discrete.append(rating)
            raw_scores.append(raw_score)
            raw_texts.append(raw_text)

    return discrete, raw_scores, raw_texts


def collect_input_images(
    image_paths,
    image_dir: str,
    env_id: str,
    seed: int,
    random_steps: int,
    capture_count: int,
    max_images: int,
):
    collected = []

    if image_paths:
        for path in image_paths:
            collected.append((Image.open(path).convert("RGB"), path))

    if image_dir:
        valid_ext = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)])
        for name in filenames:
            full_path = os.path.join(image_dir, name)
            collected.append((Image.open(full_path).convert("RGB"), full_path))

    if not collected:
        for idx in range(capture_count):
            frame = capture_env_frame(env_id, seed + idx, random_steps)
            image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
            source = f"{env_id} (captured #{idx + 1})"
            collected.append((image, source))

    if max_images > 0:
        collected = collected[:max_images]
    return collected


def main():
    parser = argparse.ArgumentParser(description="Quick Qwen VLM rating sanity test.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--goal", type=str, default="a pole balancing upright on a cart")
    parser.add_argument("--image-path", type=str, default=None, help="Optional path to RGB image. If omitted, captures env frame.")
    parser.add_argument("--image-paths", nargs="+", default=None, help="Optional list of image paths to compare.")
    parser.add_argument("--image-dir", type=str, default=None, help="Optional directory of images to compare.")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--random-steps", type=int, default=20)
    parser.add_argument("--capture-count", type=int, default=1, help="How many env snapshots to capture when no image path(s) are given.")
    parser.add_argument("--max-images", type=int, default=0, help="Max images to process. 0 means no limit.")
    parser.add_argument("--output-dir", type=str, default="evaluations/vlm_rating_debug")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--vlm-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vlm-temperature", type=float, default=0.2)
    parser.add_argument("--vlm-top-p", type=float, default=0.9)
    parser.add_argument("--vlm-collapse-threshold", type=float, default=0.8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    selected_paths = []
    if args.image_path is not None:
        selected_paths.append(args.image_path)
    if args.image_paths:
        selected_paths.extend(args.image_paths)

    items = collect_input_images(
        image_paths=selected_paths,
        image_dir=args.image_dir,
        env_id=args.env_id,
        seed=args.seed,
        random_steps=args.random_steps,
        capture_count=args.capture_count,
        max_images=args.max_images,
    )
    if not items:
        raise RuntimeError("No images found to rate.")

    processor, model = load_qwen_model(args.model_name, device)
    batch_images = [image for image, _ in items]
    batch_ratings, batch_raw_scores, batch_raw_texts = annotate_image_batch(
        images=batch_images,
        goal=args.goal,
        processor=processor,
        model=model,
        device=device,
        args=args,
    )

    all_results = []
    for idx, (image, source) in enumerate(items):
        input_image_path = os.path.join(args.output_dir, f"input_{stamp}_{idx:03d}.png")
        image.save(input_image_path)

        rating = batch_ratings[idx]
        raw_score = batch_raw_scores[idx]
        raw_text = batch_raw_texts[idx]

        annotated_path = os.path.join(args.output_dir, f"annotated_{stamp}_{idx:03d}.png")
        save_annotated_image(image, rating, raw_text, annotated_path)

        all_results.append(
            {
                "index": idx,
                "source": source,
                "input_image": input_image_path,
                "annotated_image": annotated_path,
                "rating": rating,
                "raw_score": raw_score,
                "description": extract_description(raw_text),
                "raw_text": raw_text,
            }
        )

    result = {
        "timestamp": stamp,
        "model_name": args.model_name,
        "device": str(device),
        "goal": args.goal,
        "num_images": len(all_results),
        "config": {
            "vlm_do_sample": args.vlm_do_sample,
            "vlm_temperature": args.vlm_temperature,
            "vlm_top_p": args.vlm_top_p,
            "vlm_collapse_threshold": args.vlm_collapse_threshold,
        },
        "items": all_results,
    }
    result_path = os.path.join(args.output_dir, f"result_{stamp}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== Qwen VLM Rating Test ===")
    print(f"Goal: {args.goal}")
    print(f"Processed images: {len(all_results)}")
    print("Comparison (index | rating | source):")
    for item in all_results:
        print(f"  {item['index']:03d} | {item['rating']}/5 | {item['source']}")
    print(f"Saved result json: {result_path}")


if __name__ == "__main__":
    main()

