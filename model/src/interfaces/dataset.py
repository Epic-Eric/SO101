import os
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from PIL import Image
import torch
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


class ImageJointSequenceDataset(Dataset):
    """Sequence dataset backed by `joints.jsonl` + corresponding images.

    Expected jsonl schema per line:
      {"t": <float>, "image": "frame_000123.jpg", "joints": {"joint_name": <float>, ...}}

    Returns:
      images:  (T,C,H,W) float tensor
      actions: (T-1,A) float tensor (derived from joints)
    """

    def __init__(
        self,
        root_dir: str,
        seq_len: int = 16,
        image_size: int = 64,
        normalize_images: bool = True,
        norm_params: Optional[NormalizationParams] = None,
        action_mode: str = "delta",  # 'delta' or 'pos'
        normalize_actions: bool = True,
        eps: float = 1e-6,
        cache_images: bool = False,
        cache_size: int = 2048,
        preload_images: bool = False,
        preload_dtype: str = "float16",
    ):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.root_dir = root_dir
        self.seq_len = int(seq_len)
        self.image_size = int(image_size)
        self.norm_params = norm_params or NormalizationParams()
        self.action_mode = action_mode
        self._cache_images = bool(cache_images)
        self._cache_size = int(cache_size)
        self._image_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._preload_images = bool(preload_images)
        self._preload_dtype = str(preload_dtype)

        joints_path = os.path.join(root_dir, "joints.jsonl")
        if not os.path.isfile(joints_path):
            raise FileNotFoundError(f"joints.jsonl not found in {root_dir}")

        records: List[Dict] = []
        with open(joints_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                img_name = rec.get("image")
                joints = rec.get("joints")
                if not isinstance(img_name, str) or not isinstance(joints, dict):
                    continue
                img_path = os.path.join(root_dir, img_name)
                if not os.path.isfile(img_path):
                    continue
                records.append({"image": img_path, "joints": joints, "t": rec.get("t", None)})

        if len(records) < self.seq_len:
            raise ValueError(f"Not enough frames for seq_len={self.seq_len}: found {len(records)} valid records")

        # Determine joint key order (sorted for stability)
        first_joints = records[0]["joints"]
        self.joint_keys = sorted(first_joints.keys())
        self.action_dim = len(self.joint_keys)

        # Build joint matrix
        joints_mat = []
        for rec in records:
            j = rec["joints"]
            joints_mat.append([float(j.get(k, 0.0)) for k in self.joint_keys])
        self._joints = torch.tensor(joints_mat, dtype=torch.float32)
        self._image_paths = [rec["image"] for rec in records]

        # Precompute per-step actions (N-1, A)
        if self.action_mode == "delta":
            actions = self._joints[1:] - self._joints[:-1]
        elif self.action_mode == "pos":
            actions = self._joints[:-1]
        else:
            raise ValueError("action_mode must be 'delta' or 'pos'")

        self._action_mean = actions.mean(dim=0)
        self._action_std = actions.std(dim=0).clamp_min(eps)
        self._normalize_actions = bool(normalize_actions)

        # Image transforms
        t = [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()]  # [0,1]
        if normalize_images:
            t.append(transforms.Normalize(self.norm_params.mean, self.norm_params.std))
        self.transform = transforms.Compose(t)

        # Optional: preload all transformed frames into RAM once.
        # This is especially helpful on Colab when data is on Google Drive.
        self._preloaded: Optional[torch.Tensor] = None
        if self._preload_images:
            dtype = torch.float16 if self._preload_dtype.lower() in ("float16", "fp16") else torch.float32
            imgs_all: List[torch.Tensor] = []
            for p in self._image_paths:
                img = Image.open(p).convert("RGB")
                imgs_all.append(self.transform(img).to(dtype=dtype))
            # (N,C,H,W)
            self._preloaded = torch.stack(imgs_all, dim=0)

    def _get_image(self, path: str) -> torch.Tensor:
        if self._preloaded is not None:
            raise RuntimeError("_get_image should not be called when preload_images is enabled")
        if not self._cache_images:
            img = Image.open(path).convert("RGB")
            return self.transform(img)

        # LRU cache of transformed tensors
        cached = self._image_cache.get(path)
        if cached is not None:
            self._image_cache.move_to_end(path)
            return cached

        img = Image.open(path).convert("RGB")
        t = self.transform(img)
        self._image_cache[path] = t
        self._image_cache.move_to_end(path)
        if self._cache_size > 0 and len(self._image_cache) > self._cache_size:
            self._image_cache.popitem(last=False)
        return t

    def __len__(self) -> int:
        return len(self._image_paths) - self.seq_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(idx)
        end = start + self.seq_len

        if self._preloaded is not None:
            images = self._preloaded[start:end]
            # ensure float32 downstream unless caller wants fp16
            if images.dtype != torch.float32:
                images = images.float()
        else:
            imgs = []
            for p in self._image_paths[start:end]:
                imgs.append(self._get_image(p))
            images = torch.stack(imgs, dim=0)  # (T,C,H,W)

        # actions for transitions: (T-1, A)
        actions = None
        if self.action_mode == "delta":
            a = (self._joints[start + 1 : end] - self._joints[start : end - 1])
        else:  # pos
            a = self._joints[start : end - 1]
        if self._normalize_actions:
            a = (a - self._action_mean) / self._action_std
        actions = a
        return images, actions
