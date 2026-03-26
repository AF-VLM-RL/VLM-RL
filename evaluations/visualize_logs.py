#!/usr/bin/env python3
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values[:]

    out = []
    half = window // 2
    n = len(values)

    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        out.append(sum(values[left:right]) / (right - left))

    return out


def parse_log_file(log_path):
    pattern = re.compile(
        r"global_step\s*=\s*(\d+)\s*,\s*episodic_return\s*=\s*\[?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]?"
    )

    steps = []
    rewards = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                rewards.append(float(match.group(2)))

    return steps, rewards


def make_plot(steps, rewards, output_path, smooth_window=15, title=None):
    if not steps:
        raise ValueError("No global_step / episodic_return pairs found in the log.")

    smoothed = moving_average(rewards, smooth_window)

    plt.figure(figsize=(12, 7), dpi=180)

    # Raw curve
    plt.plot(steps, rewards, alpha=0.35, linewidth=1.2, label="Raw episodic return")

    # Smoothed curve
    plt.plot(steps, smoothed, linewidth=2.5, label=f"Smoothed episodic return (window={smooth_window})")

    plt.xlabel("global_step", fontsize=12)
    plt.ylabel("episode_reward", fontsize=12)
    plt.title(title or "Episode Reward vs Global Step", fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Plot global_step vs episodic_return from a training log."
    )
    parser.add_argument("input_file", help="Path to log file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image filename (default: images/<input_filename>.png)",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=15,
        help="Moving average window size",
    )
    parser.add_argument(
        "--title",
        default="Training Curve",
        help="Plot title",
    )
    args = parser.parse_args()

    if args.output is None:
        stem = Path(args.input_file).stem
        args.output = f"images/{stem}.png"

    steps, rewards = parse_log_file(args.input_file)
    make_plot(
        steps=steps,
        rewards=rewards,
        output_path=args.output,
        smooth_window=args.window,
        title=args.title,
    )

    print(f"Parsed {len(steps)} points.")
    print(f"Saved plot to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()