import collections
import gymnasium as gym
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, text_goal, device, model_id="Qwen/Qwen2-VL-2B-Instruct",
                 n_frames=4, frame_every=4, clip_every=16, fps=1,
                 delta_reward=True, normalize_reward=True, reward_scale=10.0):
        assert clip_every >= frame_every, "clip_every must be >= frame_every"
        assert clip_every % frame_every == 0, "clip_every must be a multiple of frame_every"

        super().__init__(env)
        self.device = device
        self.n_frames = n_frames
        self.frame_every = frame_every
        self.clip_every = clip_every
        self.fps = fps
        self.step_count = 0
        self.last_reward = 0.0

        self.delta_reward = delta_reward
        self.prev_raw_score = None

        self.normalize_reward = normalize_reward
        self.reward_scale = reward_scale
        self._reward_running_mean = 0.0
        self._reward_running_var = 1.0
        self._reward_count = 0

        self.frame_buffer = collections.deque(maxlen=n_frames)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.text_goal = text_goal

        self.yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]

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
            {"type": "video", "video": pil_frames, "fps": self.fps},
            {"type": "text", "text": prompt},
        ]

        return content

    def _get_description(self, pil_frames, prompt):
        messages = [
            {
                "role": "user",
                "content": self._build_prompt(pil_frames, prompt),
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=500)

        input_len = inputs["input_ids"].shape[1]
        description = self.processor.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )
        return description

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
        self.last_reward = 0.0
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
            self.last_reward = self._compute_reward(raw_score)
            print(f"[VLM Debug] step={self.step_count} | "
                  f"raw={raw_score:.4f} | shaped={self.last_reward:.4f} | "
                  f"reward_mean={self._reward_running_mean:.4f}")

        info["vlm_reward"] = self.last_reward
        return obs, self.last_reward, terminated, truncated, info

    def compute_vlm_reward(self):
        pil_frames = [
            Image.fromarray(np.ascontiguousarray(f).astype(np.uint8))
            for f in self.frame_buffer
        ]

        # description = self._get_description(
        #     pil_frames,
        #     f"These are {self.n_frames} consecutive frames of a robot in a physics simulation. "
        #     f"Describe what the robot is doing across these frames."
        # )
        # print(f"[VLM Description] {description}")

        messages = [
            {
                "role": "user",
                "content": self._build_prompt(
                    pil_frames,
                    f"Does this show {self.text_goal}? "
                    f"Answer only Yes or No."
                ),
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_logits = outputs.logits[0, -1, :]       # (vocab_size,)
        yes_no_logits = last_logits[[self.yes_token_id, self.no_token_id]]
        probs = torch.softmax(yes_no_logits, dim=0)
        reward = probs[0].item()                     # P("Yes"), in [0, 1]

        return reward

# Improvements so far:
# 1. Added passing multiple frames to Qwen
# 2. Added option for delta reward (delta reward = current score - previous score)
# 3. Added running mean and std normalization for rewards
# 4. Set reward value as 0 when not computing Qwen score (might be removed)
# 5. Added chain-of thought prompting