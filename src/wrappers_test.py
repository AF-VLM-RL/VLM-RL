import collections
import os
import gymnasium as gym
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, text_goal, device, n_frames=4, frame_every=4, clip_every=16,
                 model_id="Qwen/Qwen2-VL-2B-Instruct",
                 delta_reward=True, normalize_reward=True, reward_scale=10.0,
                 save_dir="test"):
        assert clip_every >= frame_every, "clip_every must be >= frame_every"
        assert clip_every % frame_every == 0, "clip_every must be a multiple of frame_every"

        super().__init__(env)
        self.device = device
        self.n_frames = n_frames
        self.frame_every = frame_every
        self.clip_every = clip_every
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

        self.save_dir = save_dir
        self._save_count = 0

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
    
    def _get_description(self, pil_image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
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
            output_ids = self.model.generate(**inputs, max_new_tokens=200)

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
        print(f"[VLM Debug] Reset raw score: {raw_score:.4f}")
        return obs, info

    def step(self, action):
        obs, _original_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        if self.step_count % self.frame_every == 0:
            frame = self.env.render()
            assert frame is not None, "FATAL: env.render() returned None during step()!"
            self.frame_buffer.append(frame)

        if self.step_count % self.clip_every == 0:
            raw_score = self.compute_vlm_reward()
            self.last_reward = self._compute_reward(raw_score)
            print(f"[VLM Debug] Step {self.step_count} | "
                  f"raw={raw_score:.4f} | shaped={self.last_reward:.4f} | "
                  f"reward_mean={self._reward_running_mean:.4f}")
        # else:
        #     self.last_reward = 0.0

        info["vlm_reward"] = self.last_reward
        return obs, self.last_reward, terminated, truncated, info

    def compute_vlm_reward(self):
        combined = np.concatenate(list(self.frame_buffer), axis=1)  # (H, W*N, C)
        combined = np.ascontiguousarray(combined)
        pil_image = Image.fromarray(combined.astype(np.uint8))

        save_path = os.path.join(self.save_dir, f"frame_{self._save_count:05d}.png")
        pil_image.save(save_path)
        self._save_count += 1

        description_info = self._get_description(pil_image, "Describe what the Ant robot is doing in detail across these frames")
        print(f"[VLM Description] {description_info}")
        description_answer = self._get_description(pil_image, f"Does this show {self.text_goal}? Answer only Yes or No.")
        print(f"[VLM Description] {description_answer}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": (
                        f"You observed: \"{description_info}\"\n",
                        f"Does this show {self.text_goal}? "
                        f"Answer only Yes or No."
                    )},
                ],
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

        last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
        yes_no_logits = last_logits[[self.yes_token_id, self.no_token_id]]
        probs = torch.softmax(yes_no_logits, dim=0)
        reward = probs[0].item()  # P("Yes"), in [0, 1]

        return reward