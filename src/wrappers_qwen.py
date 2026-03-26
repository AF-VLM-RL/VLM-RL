import collections
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, text_goal, device, n_frames=4, frame_every=4, clip_every=16,
                 model_id="Qwen/Qwen2-VL-2B-Instruct"):
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

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.prompt_text = (
            "Does this image show {goal}? "
            "Answer only Yes or No.".format(goal=text_goal)
        )

        self.yes_token_id = self.processor.tokenizer.encode(
            "Yes", add_special_tokens=False
        )[0]
        self.no_token_id = self.processor.tokenizer.encode(
            "No", add_special_tokens=False
        )[0]

        print("Qwen-VL Initialized: model={0}, n_frames={1}, frame_every={2}, clip_every={3}".format(
            model_id, n_frames, frame_every, clip_every
        ))
        print("Prompt: '{0}'".format(self.prompt_text))
        print("Yes token id: {0}, No token id: {1}".format(self.yes_token_id, self.no_token_id))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0

        frame = self.env.render()
        assert frame is not None, "FATAL: env.render() returned None! Check render_mode='rgb_array'."

        self.frame_buffer.clear()
        for _ in range(self.n_frames):
            self.frame_buffer.append(frame)

        self.last_reward = self.compute_vlm_reward()
        print("[VLM Debug] Reset Reward: {0:.4f}".format(self.last_reward))

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
            print("[VLM Debug] Step {0} Reward: {1:.4f}".format(
                self.step_count, self.last_reward
            ))

        info["vlm_reward"] = self.last_reward
        return obs, self.last_reward, terminated, truncated, info

    def compute_vlm_reward(self):
        combined = np.concatenate(list(self.frame_buffer), axis=1)  # (H, W*N, C)
        combined = np.ascontiguousarray(combined)
        pil_image = Image.fromarray(combined.astype(np.uint8))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text",  "text": self.prompt_text},
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
        reward = probs[0].item()  # probability of "Yes", in [0, 1]

        return reward