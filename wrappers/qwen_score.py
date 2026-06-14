import collections
import gymnasium as gym
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, text_goal, device, model_id="Qwen/Qwen3-VL-8B-Instruct",
                 n_frames=4, frame_every=4, clip_every=16,
                 delta_reward=False, normalize_reward=False, reward_scale=10.0,
                 cot=True):
        assert clip_every >= frame_every, "clip_every must be >= frame_every"
        assert clip_every % frame_every == 0, "clip_every must be a multiple of frame_every"

        super().__init__(env)
        self.device = device
        self.cot = cot

        self.n_frames = n_frames
        self.frame_every = frame_every
        self.clip_every = clip_every

        self.step_count = 0
        self.prev_reward = 0.0
        self.prev_raw_score = None

        self.delta_reward = delta_reward
        self.normalize_reward = normalize_reward
        self.reward_scale = reward_scale

        self._reward_running_mean = 0.0
        self._reward_running_var = 1.0
        self._reward_count = 0

        self.frame_buffer = collections.deque(maxlen=n_frames)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.text_goal = text_goal

        print("Qwen-VL Initialized: model={0}, n_frames={1}, frame_every={2}, clip_every={3}".format(
            model_id, n_frames, frame_every, clip_every
        ))
        print("delta_reward={0}, normalize_reward={1}, reward_scale={2}".format(
            delta_reward, normalize_reward, reward_scale
        ))

    def _update_reward(self, reward):
        self._reward_count += 1
        delta = reward - self._reward_running_mean
        self._reward_running_mean += delta / self._reward_count
        delta2 = reward - self._reward_running_mean
        self._reward_running_var += (delta * delta2 - self._reward_running_var) / self._reward_count

    def _normalize_reward(self, reward):
        std = max(np.sqrt(self._reward_running_var), 1e-8)
        return self.reward_scale * (reward - self._reward_running_mean) / std

    def _compute_reward(self, raw_score):
        if self.delta_reward and self.prev_raw_score is not None:
            reward = raw_score - self.prev_raw_score
        else:
            reward = raw_score

        self.prev_raw_score = raw_score

        if self.normalize_reward:
            self._update_reward(reward)
            reward = self._normalize_reward(reward)

        return reward

    def _build_prompt(self, pil_frames, prompt):
        content = [
            {"type": "video", "video": pil_frames},
            {"type": "text", "text": prompt},
        ]
        return content

    def _generate_text(self, pil_frames, prompt, max_new_tokens=500):
        messages = [
            {
                "role": "user",
                "content": self._build_prompt(pil_frames, prompt),
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            num_frames=self.n_frames,
            fps=None,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        input_len = inputs["input_ids"].shape[1]
        description = self.processor.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )
        return description.strip()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0

        frame = self.env.render()
        assert frame is not None, "FATAL: env.render() returned None."

        self.frame_buffer.clear()
        for _ in range(self.n_frames):
            self.frame_buffer.append(frame)

        raw_score = self.compute_vlm_reward()
        self.prev_raw_score = raw_score
        self.prev_reward = 0.0
        print(f"[VLM Debug] Reset: raw={raw_score:.4f}")
        return obs, info

    def step(self, action):
        obs, _original_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        if self.step_count % self.frame_every == 0:
            frame = self.env.render()
            assert frame is not None, "FATAL: env.render() returned None."
            self.frame_buffer.append(frame)

        if self.step_count % self.clip_every == 0:
            raw_score = self.compute_vlm_reward()
            self.prev_reward = self._compute_reward(raw_score)
            print(f"[VLM Debug] step={self.step_count} | "
                  f"raw={raw_score:.4f} | shaped={self.prev_reward:.4f} | "
                  f"reward_mean={self._reward_running_mean:.4f}")

        info["vlm_reward"] = self.prev_reward
        return obs, self.prev_reward, terminated, truncated, info

    def compute_vlm_reward(self):
        pil_frames = [
            Image.fromarray(np.ascontiguousarray(f).astype(np.uint8))
            for f in self.frame_buffer
        ]

        local_goal = (
            "A four-legged ant-like robot moving forward to the right quickly and stably across the ground, "
            "using coordinated leg motion without flipping over, spinning, sliding, or dragging its body."
        )
        prompt_template = (
            "{header}\n"
            f"Goal: {local_goal}\n\n"
            "Rate how well the robot is achieving the goal on a scale from 0 to 10.\n\n"
            "Scoring guide:\n"
            "  0 = flipped over, completely still, stuck, or moving backward\n"
            "  2 = mostly unstable, spinning in place, dragging its body, or barely moving\n"
            "  4 = some forward movement, but mostly sliding, tumbling, twisting, or using its legs poorly\n"
            "  6 = moving forward, but awkwardly, slowly, or with poor body stability\n"
            "  8 = moving forward reasonably well with mostly stable body posture and leg motion\n"
            " 10 = smooth, stable, fast forward movement using coordinated four-legged walking\n\n"
            "Only judge visible behavior. Prefer stable forward movement with coordinated leg motion over spinning, sliding, dragging, tumbling, or falling forward.\n"
            "Respond with only a single integer from 0 to 10."
        )

        if self.cot:
            description = self._generate_text(
                pil_frames,
                "Watch this video of a four-legged ant-like robot in a physics simulation. "
                "In one sentence, describe the visible motion of the robot. "
                "Mention whether it is upright, moving forward, moving backward, spinning, "
                "flipping over, dragging its body, sliding, stuck, or using its legs to walk."
            )
            header = (
                "The following video shows a four-legged ant-like robot in a physics simulation.\n"
                "A preliminary visual description is:\n"
                f"{description}\n"
                "Use the video itself as the main evidence. If the description conflicts with the video, ignore the description.\n"
            )
        else:
            header = "The following video shows a four-legged ant-like robot in a physics simulation.\n"

        prompt = prompt_template.format(header=header)
        raw_text = self._generate_text(pil_frames, prompt, max_new_tokens=8)

        if self.cot:
            print(f"[VLM Score] Description: {description}\n    -> Score: '{raw_text}'")
        else:
            print(f"[VLM Score] Score: '{raw_text}'")

        try:
            score = int(raw_text.split()[0])
            score = max(0, min(10, score))  # clamp to [0, 10]
        except (ValueError, IndexError):
            print(f"[VLM Warning] Could not parse score from: '{raw_text}', defaulting to 5")
            score = 5

        return score / 10.0  # normalize to [0, 1]

# Improvements so far:
# 1. Added passing multiple frames to Qwen
# 2. Added option for delta reward (delta reward = current score - previous score)
# 3. Added running mean and std normalization for rewards
# 4. Set reward value as 0 when not computing Qwen score (might be removed)
# 5. Added Chain-of-Thought prompting
# 6. Upgraded to Qwen3-VL
# 7. Switched from binary Yes/No logit to numeric 0-10 generation
