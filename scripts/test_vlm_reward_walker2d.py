import os
import sys
import time
import argparse

import gymnasium as gym
import numpy as np
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.wrappers import FrameBufferWrapper, BatchedVLMRewardVecWrapper  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Probe VLM rewards on Walker2d-v4 by logging per-step VLM scores."
    )
    parser.add_argument("--env-id", type=str, default="Walker2d-v4")
    parser.add_argument(
        "--vlm-goal",
        type=str,
        default=(
            "a 2D robot with a small central torso and two jointed legs but no arms, "
            "staying upright and walking steadily to the right over time without falling"
        ),
        help="Natural language goal passed to the VLM reward wrapper.",
    )
    parser.add_argument(
        "--vlm-model-name",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Hugging Face model id for the VLM.",
    )
    parser.add_argument(
        "--vlm-max-frames",
        type=int,
        default=16,
        help="Number of recent frames to buffer for the VLM sequence reward.",
    )
    parser.add_argument(
        "--vlm-layout",
        type=str,
        default="grid",
        choices=["strip", "grid"],
        help="How to compose buffered frames for the VLM ('strip' or 'grid').",
    )
    parser.add_argument(
        "--vlm-grid-size",
        type=int,
        default=4,
        help="Grid size when vlm_layout='grid' (4 -> 4x4 = 16 frames).",
    )
    parser.add_argument(
        "--vlm-device",
        type=str,
        default=None,
        help="Device for the VLM (e.g. 'cuda', 'cpu'). Defaults to same as PPO/device map.",
    )
    parser.add_argument(
        "--vlm-max-new-tokens",
        type=int,
        default=24,
        help="Maximum number of new tokens Qwen can generate per VLM call.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to record for VLM reward probing.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=1000,
        help="Maximum environment steps per episode (to avoid extremely long rollouts).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="vlm_reward_probe",
        help="Root directory where frames and rewards will be saved.",
    )

    args = parser.parse_args()

    timestamp = int(time.time())
    run_dir = os.path.join(
        args.out_dir, f"walker2d_vlm_probe_{timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)

    import torch

    device = torch.device(args.vlm_device) if args.vlm_device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Build a single-environment vectorised Walker2d with the same wrappers
    # used in ppo_continuous_action_vlm (FrameBufferWrapper + BatchedVLMRewardVecWrapper).
    def make_env():
        env = gym.make(args.env_id, render_mode="rgb_array")
        env = gym.wrappers.FlattenObservation(env)
        env = FrameBufferWrapper(
            env,
            max_frames=args.vlm_max_frames,
            layout=args.vlm_layout,
            grid_size=args.vlm_grid_size,
            frame_stride=1,
        )
        return env

    # Vector env with one environment, to match the PPO experiment wrapper stack.
    envs = gym.vector.SyncVectorEnv([make_env])
    envs = BatchedVLMRewardVecWrapper(
        envs,
        goal=args.vlm_goal,
        device=device,
        model_name=args.vlm_model_name,
        skip_frames=1,  # ignored; wrapper recomputes every step
        reward_scale=1.0,
        layout=args.vlm_layout,
        max_new_tokens=args.vlm_max_new_tokens,
    )

    print(f"Running Walker2d VLM reward probe. Saving to: {run_dir}")

    for ep in range(args.num_episodes):
        episode_dir = os.path.join(run_dir, f"episode_{ep}")
        os.makedirs(episode_dir, exist_ok=True)
        rewards_path = os.path.join(episode_dir, "vlm_rewards.txt")
        texts_path = os.path.join(episode_dir, "vlm_raw_texts.txt")

        obs, info = envs.reset()
        done = False
        step = 0
        episodic_return = 0.0

        with open(rewards_path, "w") as f_rewards, open(texts_path, "w") as f_texts:
            f_rewards.write("# step\tvlm_reward\n")
            f_texts.write("# step\tvlm_text\n")

            while not done and step < args.max_steps_per_episode:
                # Random actions are enough to probe the VLM's scoring behaviour.
                action = envs.single_action_space.sample()
                # SyncVectorEnv expects a batch of actions, shape (num_envs, action_dim)
                obs, reward, terminated, truncated, info = envs.step(
                    np.expand_dims(action, axis=0)
                )
                done = bool(terminated[0] or truncated[0])

                # BatchedVLMRewardVecWrapper returns the VLM reward as the env reward.
                vlm_reward = float(reward[0])
                episodic_return += vlm_reward

                # Save composite image the VLM sees (frame grid/strip).
                composite = envs.call("get_composite_image")[0]
                frame_path = os.path.join(episode_dir, f"step_{step:05d}.png")
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                if isinstance(composite, Image.Image):
                    composite.save(frame_path)
                else:
                    Image.fromarray(composite).save(frame_path)

                f_rewards.write(f"{step}\t{vlm_reward:.6f}\n")

                # Save raw VLM text output for this step.
                raw_texts = envs.get_wrapper_attr("last_texts")
                vlm_text = (
                    raw_texts[0]
                    if isinstance(raw_texts, list) and raw_texts
                    else raw_texts
                )
                safe_text = str(vlm_text).replace("\n", " ").replace("\t", " ")
                f_texts.write(f"{step}\t{safe_text}\n")

                step += 1

        print(f"Episode {ep}: steps={step}, sum_vlm_reward={episodic_return:.4f}")

    envs.close()
    print("Done running Walker2d VLM reward probe.")


if __name__ == "__main__":
    main()

