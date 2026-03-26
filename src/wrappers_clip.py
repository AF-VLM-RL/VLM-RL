import collections
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from transformers import CLIPModel, CLIPTokenizer


class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, text_goal, device, n_frames=4, frame_every=4, clip_every=16):
        """
        n_frames    : how many frames to concatenate and show CLIP
        frame_every : collect one frame into the buffer every K steps
        clip_every  : run CLIP every M steps (must be >= frame_every)
        """
        assert clip_every >= frame_every, "clip_every must be >= frame_every"
        assert clip_every % frame_every == 0, "clip_every must be a multiple of frame_every"

        super().__init__(env)
        self.device = device
        self.n_frames = n_frames
        self.frame_every = frame_every
        self.clip_every = clip_every
        self.step_count = 0
        self.last_reward = 0.0

        self.frame_buffer = collections.deque(maxlen=n_frames)

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).half()
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        with torch.no_grad():
            text_inputs = tokenizer([text_goal], padding=True, return_tensors="pt")
            text_feats = self.model.get_text_features(**text_inputs.to(device))
            self.text_features = (text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)).half()

        self.mean = torch.tensor([0.4814, 0.4578, 0.4082], device=device, dtype=torch.float16).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.2686, 0.2613, 0.2757], device=device, dtype=torch.float16).view(1, 3, 1, 1)

        print(f"VLM Initialized: n_frames={n_frames}, frame_every={frame_every}, clip_every={clip_every}, goal='{text_goal}'")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0

        frame = self.env.render()
        assert frame is not None, "FATAL: env.render() returned None! Check render_mode='rgb_array'."

        self.frame_buffer.clear()
        for _ in range(self.n_frames):
            self.frame_buffer.append(frame)

        self.last_reward = self.compute_vlm_reward()
        print(f"[VLM Debug] Reset Reward: {self.last_reward:.4f}")

        return obs, info

    def step(self, action):
        obs, _original_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        if self.step_count % self.frame_every == 0:
            frame = self.env.render()
            assert frame is not None, "FATAL: env.render() returned None during step()!"
            self.frame_buffer.append(frame)

        if self.step_count % self.clip_every == 0:
            self.last_reward = self.compute_vlm_reward()
            print(f"[VLM Debug] Step {self.step_count} Reward: {self.last_reward:.4f}")

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