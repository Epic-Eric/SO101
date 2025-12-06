import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class NormalizationParams:
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)


class ImageFolder64Dataset(Dataset):
    """64x64 RGB dataset with optional normalization.

    Stores normalization parameters for downstream reference/visualization.
    """

    def __init__(self, root_dir: str, normalize: bool = True, norm_params: Optional[NormalizationParams] = None):
        self.root_dir = root_dir
        self.paths: List[str] = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(root_dir)
            for f in filenames
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.norm_params = norm_params or NormalizationParams()

        t = [transforms.Resize((64, 64)), transforms.ToTensor()]  # [0,1]
        if normalize:
            t.append(transforms.Normalize(self.norm_params.mean, self.norm_params.std))
        self.transform = transforms.Compose(t)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)
