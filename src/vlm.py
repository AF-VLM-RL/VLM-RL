import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPModel, CLIPTokenizer
import numpy as np

class BatchedVLM:
    def __init__(self, text_goal, device):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Pre-compute text features (Goal is constant)
        with torch.no_grad():
            text_inputs = self.tokenizer([text_goal], padding=True, return_tensors="pt")
            self.text_features = self.model.get_text_features(**text_inputs.to(device))
            self.text_features /= self.text_features.norm(p=2, dim=-1, keepdim=True)

        self.clip_transform = T.Compose([
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                        std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def get_rewards(self, frame_list):
        """
        Args:
            frame_list: List of numpy arrays [(H, W, 3), (H, W, 3), ...]
        Returns:
            Tensor of rewards [B]
        """
        # 1. Stack numpy arrays into one tensor (B, H, W, 3)
        # This is faster than moving them one by one
        batch_np = np.stack(frame_list) 
        
        # 2. Move to GPU once (B, H, W, 3) -> (B, 3, H, W)
        img_tensor = torch.from_numpy(batch_np).to(self.device, dtype=torch.float32)
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        
        # 3. Batch Resize & Normalize
        img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        img_tensor = img_tensor / 255.0
        img_tensor = self.clip_transform(img_tensor)

        # 4. Batched Inference (One GPU call for N envs)
        with torch.no_grad():
            image_features = self.model.get_image_features(img_tensor)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            
            # (B, 512) @ (512, 1) -> (B, 1)
            rewards = (image_features @ self.text_features.T).squeeze(1)
            
        return rewards.cpu().numpy() # Return to CPU for PPO storage