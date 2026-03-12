import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from transformers import CLIPModel, CLIPTokenizer

class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, text_goal, device, skip_frames=8):
        super().__init__(env)
        self.device = device
        self.skip_frames = skip_frames
        self.step_count = 0
        self.last_reward = 0.0
        
        # 1. Load Model in HALF PRECISION (FP16) - Massive Speedup!
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).half()
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # 2. Pre-compute Text Features (FP16)
        with torch.no_grad():
            text_inputs = tokenizer([text_goal], padding=True, return_tensors="pt")
            text_feats = self.model.get_text_features(**text_inputs.to(device))
            # Store normalized features in FP16
            self.text_features = (text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)).half()

        # 3. Pre-calculate Normalization constants (FP16)
        # CLIP mean/std: [0.481, 0.457, 0.408] / [0.268, 0.261, 0.275]
        self.mean = torch.tensor([0.4814, 0.4578, 0.4082], device=device, dtype=torch.float16).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2686, 0.2613, 0.2757], device=device, dtype=torch.float16).view(1, 3, 1, 1)

        print(f"Turbo VLM Initialized: FP16=True, Skip={skip_frames}, Goal='{text_goal}'")
        print(f"Text Features (FP16): {self.text_features.cpu().numpy()}")
        print(f"Device: {device}, Model dtype: {self.model.parameters().__next__().dtype}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        
        frame = self.env.render()
        
        # LOUD ERROR if rendering fails
        assert frame is not None, "FATAL: env.render() returned None! Check render_mode='rgb_array' in make_env."
        
        self.last_reward = self.compute_vlm_reward(frame)
        print(f"[VLM Debug] Reset Reward Calculated: {self.last_reward:.4f}")
            
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        if self.step_count % self.skip_frames == 0:
            frame = self.env.render()
            assert frame is not None, "FATAL: env.render() returned None during step()!"
            
            self.last_reward = self.compute_vlm_reward(frame)
            print(f"[VLM Debug] Step {self.step_count} Reward: {self.last_reward:.4f}")

        info["vlm_reward"] = self.last_reward
        
        return obs, self.last_reward, terminated, truncated, info

    def compute_vlm_reward(self, frame_array):
        # Make sure the numpy array has positive, contiguous strides
        frame_array = np.asarray(frame_array)
        frame_array = np.ascontiguousarray(frame_array)

        # (H, W, C) -> torch tensor
        img = torch.from_numpy(frame_array).to(self.device, dtype=torch.float16)
        img = img.permute(2, 0, 1).unsqueeze(0)

        # Scale to [0, 1] and resize
        img = img / 255.0
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalize for CLIP
        img = (img - self.mean) / self.std

        with torch.no_grad():
            img_feats = self.model.get_image_features(img)
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            similarity = (img_feats @ self.text_features.T).item()

        return similarity