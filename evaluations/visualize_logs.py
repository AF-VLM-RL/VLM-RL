import re
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


EPISODIC_RETURN_PATTERN = re.compile(
    r"global_step=(\d+).*?episodic_return=([-+]?\d*\.?\d+)"
)


def parse_log_file(path):
    steps = []
    returns = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = EPISODIC_RETURN_PATTERN.search(line)
            if m:
                steps.append(int(m.group(1)))
                returns.append(float(m.group(2)))

    if len(steps) == 0:
        raise ValueError(f"No episodic_return lines found in {path}")

    # sort by step and deduplicate by keeping the last value for each step
    step_to_return = {}
    for s, r in zip(steps, returns):
        step_to_return[s] = r

    sorted_steps = np.array(sorted(step_to_return.keys()), dtype=int)
    sorted_returns = np.array([step_to_return[s] for s in sorted_steps], dtype=float)

    return sorted_steps, sorted_returns


def moving_average(y, window):
    if len(y) == 0:
        return y.copy()

    window = max(1, min(window, len(y)))
    kernel = np.ones(window, dtype=float) / window

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    y_pad = np.pad(y, (pad_left, pad_right), mode="edge")

    return np.convolve(y_pad, kernel, mode="valid")


def format_k(x, pos=None):
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k"
    return str(int(x))


def style_axes(ax, dark=False):
    if dark:
        fig_bg = "#111111"
        ax_bg = "#111111"
        grid_c = "#444444"
        spine_c = "#666666"
        text_c = "#eaeaea"
    else:
        fig_bg = "white"
        ax_bg = "white"
        grid_c = "#d9d9d9"
        spine_c = "#cccccc"
        text_c = "#222222"

    ax.figure.patch.set_facecolor(fig_bg)
    ax.set_facecolor(ax_bg)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(spine_c)

    ax.tick_params(axis="both", colors=text_c, labelsize=11)
    ax.grid(True, axis="y", color=grid_c, alpha=0.55, linewidth=0.8)
    ax.grid(False, axis="x")
    ax.title.set_color(text_c)
    return text_c


def plot_run(
    ax,
    steps,
    values,
    label,
    smooth_window_main,
    clip_percentile=None,
    smooth_window_bg=15,
):
    plot_values = values.astype(float).copy()

    if clip_percentile is not None and 0 < clip_percentile < 100:
        cap = np.percentile(plot_values, clip_percentile)
        plot_values = np.minimum(plot_values, cap)

    smooth_main = moving_average(plot_values, smooth_window_main)
    main_line, = ax.plot(
        steps,
        smooth_main,
        alpha=0.93,
        linewidth=2.3,
        label=label,
        zorder=3,
    )

    smooth_bg = moving_average(plot_values, smooth_window_bg)
    ax.plot(
        steps,
        smooth_bg,
        color=main_line.get_color(),
        alpha=0.18,
        linewidth=1.4,
        zorder=2,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log1",
        help="Path to first log file",
        default="../logs/sac_mujoco_10581633.out",
    )
    parser.add_argument(
        "--log2",
        help="Path to second log file",
        default="../logs/sac_mujoco_10581394.out",
    )
    parser.add_argument("--label1", default="Without CoT", help="Label for first run")
    parser.add_argument("--label2", default="With CoT", help="Label for second run")
    parser.add_argument("--smooth-window", type=int, default=800, help="Smoothing window size in steps")
    parser.add_argument("--smooth-window-bg", type=int, default=15, help="Background smoothing window size")
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="Visually cap raw spikes at this percentile; use 0 to disable",
    )
    parser.add_argument("--title", default="charts/episodic_return")
    parser.add_argument("--save", default="episodic_return_compare.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--dark", action="store_true")
    args = parser.parse_args()

    steps1, returns1 = parse_log_file(args.log1)
    steps2, returns2 = parse_log_file(args.log2)

    label1 = args.label1 or os.path.splitext(os.path.basename(args.log1))[0]
    label2 = args.label2 or os.path.splitext(os.path.basename(args.log2))[0]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    text_c = style_axes(ax, dark=args.dark)

    clip_percentile = None if args.clip_percentile <= 0 else args.clip_percentile

    plot_run(ax, steps1, returns1, label1, args.smooth_window, clip_percentile, args.smooth_window_bg)
    plot_run(ax, steps2, returns2, label2, args.smooth_window, clip_percentile, args.smooth_window_bg)

    ax.set_title(args.title, fontsize=18, pad=16)
    ax.set_ylim(bottom=0)

    # x-axis formatting
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_k))

    # labels placed like W&B style
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.text(
        0.995,
        0.02,
        "global_step",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color=text_c,
        alpha=0.9,
    )

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
        fontsize=12,
        handlelength=0.7,
        columnspacing=1.0,
        handletextpad=0.6,
    )
    for t in legend.get_texts():
        t.set_color(text_c)

    plt.tight_layout()
    plt.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()