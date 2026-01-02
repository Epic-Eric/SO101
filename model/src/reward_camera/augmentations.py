import cv2
import numpy as np
import torch
import random
from typing import Tuple

# OpenCV can spawn many threads per process; with DataLoader workers this can
# oversubscribe CPU badly and slow training.
try:
    cv2.setNumThreads(0)
except Exception:
    pass

class Augmenter:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: np.ndarray, heatmap: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to image and heatmap.
        Image: HxWxC (RGB, uint8)
        Heatmap: HxW (float32)
        Returns: Tensors (C, H, W) normalized
        """
        
        # Convert to float for processing
        img_aug = image.astype(np.float32) / 255.0
        hm_aug = heatmap.copy()
        
        if random.random() < self.prob:
            # 1. Geometric Transforms (Applied to both)
            h, w = img_aug.shape[:2]
            
            # Rotation
            angle = random.uniform(-5, 5)
            M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img_aug = cv2.warpAffine(
                img_aug,
                M_rot,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            hm_aug = cv2.warpAffine(
                hm_aug,
                M_rot,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            
            # Scale Jitter (Zoom in/out slightly)
            scale = random.uniform(0.9, 1.1)
            if scale != 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img_aug, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                hm_resized = cv2.resize(hm_aug, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Crop or Pad to original size
                if scale > 1.0: # Crop center
                    start_x = (new_w - w) // 2
                    start_y = (new_h - h) // 2
                    img_aug = img_resized[start_y:start_y+h, start_x:start_x+w]
                    hm_aug = hm_resized[start_y:start_y+h, start_x:start_x+w]
                else: # Pad
                    pad_x = (w - new_w) // 2
                    pad_y = (h - new_h) // 2
                    img_aug = cv2.copyMakeBorder(
                        img_resized,
                        pad_y,
                        h - new_h - pad_y,
                        pad_x,
                        w - new_w - pad_x,
                        cv2.BORDER_CONSTANT,
                        value=0,
                    )
                    hm_aug = cv2.copyMakeBorder(
                        hm_resized,
                        pad_y,
                        h - new_h - pad_y,
                        pad_x,
                        w - new_w - pad_x,
                        cv2.BORDER_CONSTANT,
                        value=0,
                    )

            # 2. Photometric Transforms (Image only)
            
            # Brightness
            brightness = random.uniform(0.8, 1.2)
            img_aug = img_aug * brightness
            
            # Contrast
            contrast = random.uniform(0.8, 1.2)
            img_aug = (img_aug - 0.5) * contrast + 0.5
            
            # Noise
            noise = np.random.normal(0, 0.02, img_aug.shape)
            img_aug = img_aug + noise
            
            # Blur
            if random.random() < 0.3:
                ksize = random.choice([3, 5])
                img_aug = cv2.GaussianBlur(img_aug, (ksize, ksize), 0)
                
            # Clip
            img_aug = np.clip(img_aug, 0, 1)

        # Keep heatmap in [0, 1] after geometric ops
        hm_max = float(np.max(hm_aug))
        if hm_max > 0:
            hm_aug = (hm_aug / hm_max).astype(np.float32)

        # Convert to Tensor
        img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float()
        hm_tensor = torch.from_numpy(hm_aug).unsqueeze(0).float()
        
        return img_tensor, hm_tensor
