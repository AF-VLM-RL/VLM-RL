import collections
import gymnasium as gym
import torch
import numpy as np
from PIL import Image
from transformers import XCLIPModel, XCLIPProcessor


class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, model_id, text_goal, device, n_frames=8, frame_every=4, clip_every=16,
                 delta_reward=True, normalize_reward=True, reward_scale=10.0):
        assert clip_every >= frame_every, "clip_every must be >= frame_every"
        assert clip_every % frame_every == 0, "clip_every must be a multiple of frame_every"
        assert n_frames == 8, "xclip requires n_frames=8"

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

        self.model = XCLIPModel.from_pretrained(model_id).to(device).half()
        self.processor = XCLIPProcessor.from_pretrained(model_id)

        with torch.no_grad():
            text_inputs = self.processor(text=[text_goal], return_tensors="pt", padding=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_feats = self.model.get_text_features(**text_inputs)
            self.text_features = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)

        print(f"X-CLIP Initialized: model={model_id} n_frames={n_frames}, frame_every={frame_every}, "
              f"clip_every={clip_every}\n",
              f"Goal: '{text_goal}', "
              f"delta_reward={delta_reward}, normalize_reward={normalize_reward}")

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

    # def compute_vlm_reward(self):
    #     frames = [np.ascontiguousarray(f).astype(np.uint8) for f in self.frame_buffer]

    #     inputs = self.processor(videos=[frames], return_tensors="pt")
    #     print(inputs)

    def compute_vlm_reward(self):
        frames = [np.ascontiguousarray(f).astype(np.uint8) for f in self.frame_buffer]
        
        inputs = self.processor(images=frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].half()

        batch_size = inputs["pixel_values"].shape[0]
        dummy_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        dummy_mask = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=dummy_ids,
                attention_mask=dummy_mask,
                pixel_values=inputs["pixel_values"],
                return_dict=True,
            )
            img_feats = outputs.video_embeds
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            similarity = (img_feats @ self.text_features.T).item()

        return similarity

# [Debug] videos=np_frames -> keys: []
# [Debug] videos=[np_frames] -> keys: []
# [Debug] videos=pil_frames -> keys: []
# [Debug] videos=[pil_frames] -> keys: []
# [Debug] images=np_frames -> keys: ['pixel_values']
# [Debug] images=[np_frames] -> keys: ['pixel_values']
# [Debug] images=pil_frames -> keys: ['pixel_values']
# [Debug] images=[pil_frames] -> keys: ['pixel_values']