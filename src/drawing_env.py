"""Hand-drawing circle environment with CLIP-based reward."""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor


class CLIPRewardModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"Loading CLIP on {device}...")
        self.device = device
        self.model_id = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_id).to(device)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)

        self.prompts = [
            "a perfect geometric circle",
            "a straight line",
            "a square",
            "a blank white page",
        ]

    def get_score(self, image_pil):
        inputs = self.processor(
            text=self.prompts, images=image_pil, return_tensors="pt", padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use raw logits instead of softmax to avoid the "vanishing gradient" 
            # at the extremes of the probability distribution
            logits = outputs.logits_per_image # Shape: [1, 4]
            
        circle_logit = logits[0, 0]
        # Compare circle to the "best" non-circle competitor
        best_competitor_logit = torch.max(logits[0, 1:])
        
        # Margin-based reward: How much 'more' like a circle is it than a square/line?
        reward = (circle_logit - best_competitor_logit).item()
        
        # Optional: Clamp or Scale to keep PPO stable (e.g., -1.0 to 1.0 range)
        return reward


class DrawingEnv(gym.Env):
    """Draw shapes on a 64x64 canvas; CLIP judges how circle-like the result is."""

    def __init__(self):
        super().__init__()
        self.canvas_size = 64
        self.max_steps = 40
        self.current_step = 0

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(2, self.canvas_size, self.canvas_size),
            dtype=np.uint8,
        )

        self.vlm = CLIPRewardModel()
        self.cursor_pos = [32, 32]
        self.canvas_history = []
        self.last_vlm_score = 0.0

    def _get_obs(self):
        img = self._render_history(self.canvas_history).convert("L")
        canvas_array = np.array(img, dtype=np.uint8)

        cursor_layer = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        cx, cy = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        cursor_layer[max(0, cy - 1) : min(64, cy + 2), max(0, cx - 1) : min(64, cx + 2)] = 255

        return np.stack([canvas_array, cursor_layer], axis=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cursor_pos = [32, 32]
        self.canvas_history = [tuple(self.cursor_pos)]
        self.last_vlm_score = 0.0
        return self._get_obs(), {}

    def step(self, action):
        step_size = 4

        if action == 0:
            self.cursor_pos[1] -= step_size
        elif action == 1:
            self.cursor_pos[1] += step_size
        elif action == 2:
            self.cursor_pos[0] -= step_size
        elif action == 3:
            self.cursor_pos[0] += step_size

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.canvas_size - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.canvas_size - 1)
        self.canvas_history.append(tuple(self.cursor_pos))
        self.current_step += 1

        terminated = self.current_step >= self.max_steps

        current_img = self._render_history(self.canvas_history)
        current_vlm_score = self.vlm.get_score(current_img)
        reward = current_vlm_score - self.last_vlm_score
        reward -= 0.01
        self.last_vlm_score = current_vlm_score

        return self._get_obs(), reward, terminated, False, {}

    def _render_history(self, history):
        img = Image.new("RGB", (self.canvas_size, self.canvas_size), color="white")
        draw = ImageDraw.Draw(img)
        if len(history) > 1:
            draw.line(history, fill="black", width=4)
        else:
            draw.point(history[0], fill="black")
        return img

    def render(self):
        return self._render_history(self.canvas_history)
