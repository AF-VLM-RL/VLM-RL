import argparse
import json
import os
from collections import Counter
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw


def load_dataset(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    if "observations" not in data or "ratings" not in data:
        raise ValueError(f"{npz_path} is missing required keys: observations, ratings")
    observations = data["observations"]
    ratings = data["ratings"].astype(np.int64)
    if len(observations) != len(ratings):
        raise ValueError(f"{npz_path} has mismatched lengths: obs={len(observations)} ratings={len(ratings)}")
    return observations, ratings


def dataset_stats(ratings: np.ndarray) -> Dict[str, float]:
    cnt = Counter(int(r) for r in ratings.tolist())
    total = len(ratings)
    out: Dict[str, float] = {"total_samples": total}
    for r in range(1, 6):
        out[f"count_{r}"] = cnt.get(r, 0)
        out[f"pct_{r}"] = (100.0 * cnt.get(r, 0) / total) if total > 0 else 0.0
    return out


def _safe_to_image(arr: np.ndarray) -> Image.Image:
    img = arr
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return Image.fromarray(img, mode="L").convert("RGB")
    if img.ndim == 3 and img.shape[-1] == 3:
        return Image.fromarray(img, mode="RGB")
    if img.ndim == 3 and img.shape[-1] == 1:
        return Image.fromarray(img.squeeze(-1), mode="L").convert("RGB")
    raise ValueError(f"Unsupported observation shape for image conversion: {img.shape}")


def save_rating_montage(
    observations: np.ndarray,
    ratings: np.ndarray,
    output_path: str,
    rating: int,
    max_images: int,
    tile_size: int = 128,
    cols: int = 8,
) -> int:
    indices = np.where(ratings == rating)[0][:max_images]
    if len(indices) == 0:
        return 0

    rows = (len(indices) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size), color=(25, 25, 25))
    for i, idx in enumerate(indices):
        img = _safe_to_image(observations[idx]).resize((tile_size, tile_size))
        x = (i % cols) * tile_size
        y = (i // cols) * tile_size
        canvas.paste(img, (x, y))
    canvas.save(output_path)
    return len(indices)


def save_count_bar_chart(counts: Dict[int, int], title: str, output_path: str) -> None:
    width, height = 720, 420
    margin = 60
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin
    bar_w = chart_w // 5 - 20
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((20, 15), title, fill=(0, 0, 0))
    max_count = max(1, max(counts.values()) if counts else 1)
    draw.line((margin, height - margin, width - margin, height - margin), fill=(0, 0, 0), width=2)
    draw.line((margin, margin, margin, height - margin), fill=(0, 0, 0), width=2)

    for i, rating in enumerate(range(1, 6)):
        count = counts.get(rating, 0)
        bar_h = int((count / max_count) * (chart_h - 20))
        x0 = margin + i * (bar_w + 20) + 10
        y0 = height - margin - bar_h
        x1 = x0 + bar_w
        y1 = height - margin
        draw.rectangle((x0, y0, x1, y1), fill=(70, 130, 180), outline=(0, 0, 0))
        draw.text((x0 + bar_w // 2 - 8, height - margin + 8), str(rating), fill=(0, 0, 0))
        draw.text((x0 + 6, y0 - 18), str(count), fill=(0, 0, 0))

    canvas.save(output_path)


def process_dataset(npz_path: str, out_dir: str, max_images_per_rating: int) -> None:
    observations, ratings = load_dataset(npz_path)
    name = os.path.splitext(os.path.basename(npz_path))[0]
    target_dir = os.path.join(out_dir, name)
    os.makedirs(target_dir, exist_ok=True)

    stats = dataset_stats(ratings)
    with open(os.path.join(target_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    counts = {r: int(stats[f"count_{r}"]) for r in range(1, 6)}
    save_count_bar_chart(counts, f"Rating Distribution: {name}", os.path.join(target_dir, "rating_distribution.png"))

    for rating in range(1, 6):
        saved = save_rating_montage(
            observations=observations,
            ratings=ratings,
            output_path=os.path.join(target_dir, f"rating_{rating}_samples.png"),
            rating=rating,
            max_images=max_images_per_rating,
        )
        print(f"[{name}] rating={rating}: saved {saved} samples")

    print(f"[{name}] total={len(ratings)} -> {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize warm-start ERL dataset npz files.")
    parser.add_argument(
        "--dqn-npz",
        type=str,
        default="runs/warmstart_datasets/dqn_erl_vlm_cartpole.npz",
        help="Path to DQN ERL warm-start dataset npz",
    )
    parser.add_argument(
        "--sac-npz",
        type=str,
        default="runs/warmstart_datasets/sac_erl_vlm_cartpole.npz",
        help="Path to SAC ERL warm-start dataset npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/warmstart_dataset_viz",
        help="Directory to write visualizations",
    )
    parser.add_argument(
        "--max-images-per-rating",
        type=int,
        default=64,
        help="Max sample images to include per rating montage",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    process_dataset(args.dqn_npz, args.output_dir, args.max_images_per_rating)
    process_dataset(args.sac_npz, args.output_dir, args.max_images_per_rating)


if __name__ == "__main__":
    main()
