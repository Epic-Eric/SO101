import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from datetime import datetime
from typing import List, Tuple, Optional

class RedBeadDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform=None,
        mode: str = 'train',
        target_hw: Optional[Tuple[int, int]] = (192, 256),
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.target_hw = target_hw
        self.samples = []
        
        os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'heatmaps'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'metadata'), exist_ok=True)
        
        self.refresh_index()

    def refresh_index(self):
        """Reloads the list of available samples."""
        self.samples = []
        img_dir = os.path.join(self.data_dir, 'images')
        if not os.path.exists(img_dir):
            return
            
        # Simple time-based split logic could be added here
        all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        # For now, just load all. Split logic can be handled by Subset in train.py
        self.samples = [f.replace('.png', '') for f in all_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        img_path = os.path.join(self.data_dir, 'images', f"{sample_id}.png")
        heatmap_path = os.path.join(self.data_dir, 'heatmaps', f"{sample_id}.npy")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        heatmap = np.load(heatmap_path)

        # Speed: train on a smaller resolution to reduce CPU augmentation and MPS compute.
        # This model is fully-convolutional, so inference can still run at full resolution.
        if self.target_hw is not None:
            th, tw = self.target_hw
            if image.shape[0] != th or image.shape[1] != tw:
                image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
            if heatmap.shape[0] != th or heatmap.shape[1] != tw:
                heatmap = cv2.resize(heatmap, (tw, th), interpolation=cv2.INTER_AREA)
                # Keep heatmap in [0, 1]
                hm_max = float(np.max(heatmap))
                if hm_max > 0:
                    heatmap = (heatmap / hm_max).astype(np.float32)
        
        if self.transform:
            # Apply transforms. Note: Heatmap needs to be transformed geometrically same as image.
            # This usually requires a library like albumentations for synchronized transforms.
            # For simplicity with torchvision, we might just do color transforms on image
            # and geometric transforms carefully or skip geometric for MVP if complex.
            # However, the prompt asks for geometric augmentation.
            # We will handle this in the augmentations module or here.
            # Let's assume transform returns (image, heatmap)
            image, heatmap = self.transform(image, heatmap)
            
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(heatmap, torch.Tensor):
            heatmap = torch.from_numpy(heatmap).unsqueeze(0).float()
            
        return image, heatmap

    def add_sample(self, image: np.ndarray, centroids: List[Tuple[int, int]], confidence: float, params: dict):
        """
        Adds a new sample to the dataset.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Generate heatmap
        h, w = image.shape[:2]
        heatmap = self.generate_gaussian_heatmap((h, w), centroids)
        
        # Save files
        cv2.imwrite(os.path.join(self.data_dir, 'images', f"{timestamp}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(os.path.join(self.data_dir, 'heatmaps', f"{timestamp}.npy"), heatmap)
        
        metadata = {
            "centroids": centroids,
            "confidence": confidence,
            "params": params,
            "timestamp": timestamp
        }
        
        with open(os.path.join(self.data_dir, 'metadata', f"{timestamp}.json"), 'w') as f:
            json.dump(metadata, f)
            
        self.samples.append(timestamp)
        return timestamp

    def delete_sample(self, sample_id: str):
        """
        Deletes a sample by its ID.
        """
        if sample_id not in self.samples:
            return False
            
        # Remove files
        img_path = os.path.join(self.data_dir, 'images', f"{sample_id}.png")
        heatmap_path = os.path.join(self.data_dir, 'heatmaps', f"{sample_id}.npy")
        metadata_path = os.path.join(self.data_dir, 'metadata', f"{sample_id}.json")
        
        for path in [img_path, heatmap_path, metadata_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # Remove from samples list
        self.samples.remove(sample_id)
        return True

    def delete_all_samples(self):
        """
        Deletes all samples.
        """
        for sample_id in self.samples[:]:  # Copy the list to avoid modification during iteration
            self.delete_sample(sample_id)
        self.samples = []

    @staticmethod
    def generate_gaussian_heatmap(shape: Tuple[int, int], centroids: List[Tuple[int, int]], sigma: float = 5.0) -> np.ndarray:
        """
        Generates a 2D Gaussian heatmap for multiple centroids.
        """
        h, w = shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        x = np.arange(0, w, 1, float)
        y = np.arange(0, h, 1, float)
        y = y[:, np.newaxis]
        
        for (cx, cy) in centroids:
            x0 = cx
            y0 = cy
            heatmap += np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
            
        # Normalize to [0, 1] if max > 1 (overlapping gaussians)
        if heatmap.max() > 1.0:
            heatmap /= heatmap.max()
            
        return heatmap
