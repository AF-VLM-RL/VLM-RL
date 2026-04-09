import collections
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from transformers import CLIPModel, CLIPTokenizer


class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, model_id, text_goal, device, n_frames=4, frame_every=4, clip_every=16,
                 delta_reward=True, normalize_reward=True, reward_scale=10.0):
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

        self.model = CLIPModel.from_pretrained(model_id).to(device).half()
        tokenizer = CLIPTokenizer.from_pretrained(model_id)

        with torch.no_grad():
            text_inputs = tokenizer([text_goal], padding=True, return_tensors="pt")
            text_feats = self.model.get_text_features(**text_inputs.to(device))
            self.text_features = (text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)).half()

        self.mean = torch.tensor([0.4814, 0.4578, 0.4082], device=device, dtype=torch.float16).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.2686, 0.2613, 0.2757], device=device, dtype=torch.float16).view(1, 3, 1, 1)

        print(f"CLIP Initialized: model={model_id} n_frames={n_frames}, frame_every={frame_every}, "
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
        # else:
        #     self.last_reward = 0.0

        info["vlm_reward"] = self.last_reward
        return obs, self.last_reward, terminated, truncated, info

    def compute_vlm_reward(self):
        combined = np.concatenate(list(self.frame_buffer), axis=1)  # (H, W*N, C)
        combined = np.ascontiguousarray(combined)

        img = torch.from_numpy(combined).to(self.device, dtype=torch.float16)
        img = img.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W*N)

        img = img / 255.0
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        img = (img - self.mean) / self.std

        with torch.no_grad():
            img_feats = self.model.get_image_features(img)
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            similarity = (img_feats @ self.text_features.T).item()

        return similarity

# Improvements so far:
# 1. Added passing multiple frames to CLIP by concatenating them
# 2. Added option for delta reward (delta reward = current score - previous score)
# 3. Added running mean and std normalization for rewards
# 4. Set reward value as 0 when not computing CLIP score
