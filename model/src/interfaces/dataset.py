import os
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


ACTION6_DIM = 6


@dataclass
class NormalizationParams:
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)


def _discover_episode_dirs(root: str) -> List[str]:
    """Return list of episode dirs containing joints.jsonl (root itself or nested)."""
    if os.path.isfile(os.path.join(root, "joints.jsonl")):
        return [root]

    # Fast-path: for the common layout `root/episode_*/joints.jsonl`, avoid an expensive
    # recursive os.walk which is very slow on networked filesystems (e.g., Colab Drive).
    try:
        immediate: List[str] = []
        for name in sorted(os.listdir(root)):
            ep_dir = os.path.join(root, name)
            if os.path.isdir(ep_dir) and os.path.isfile(os.path.join(ep_dir, "joints.jsonl")):
                immediate.append(ep_dir)
        if immediate:
            return immediate
    except FileNotFoundError:
        return []
    except Exception:
        # Fall back to recursive walk if listing fails.
        pass

    eps: List[str] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            if "joints.jsonl" in filenames:
                eps.append(dirpath)
    except FileNotFoundError:
        # Root directory missing or unreadable; treat as no episodes.
        return []
    return sorted(eps)


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
    """Episode-aware sequence dataset backed by `joints.jsonl` + corresponding images.

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

        self._normalize_actions = bool(normalize_actions)

        # Image transforms
        t = [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()]  # [0,1]
        if normalize_images:
            t.append(transforms.Normalize(self.norm_params.mean, self.norm_params.std))
        self.transform = transforms.Compose(t)

        def _load_episode(ep_dir: str) -> Optional[Dict]:
            """Load one episode folder."""
            meta_joint_keys: Optional[List[str]] = None
            meta_path = os.path.join(ep_dir, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r") as mf:
                        meta_obj = json.load(mf)
                    jk = meta_obj.get("joint_keys6") if isinstance(meta_obj, dict) else None
                    if isinstance(jk, list) and len(jk) > 0:
                        meta_joint_keys = [str(x) for x in jk][:ACTION6_DIM]
                except Exception:
                    meta_joint_keys = None

            records: List[Dict] = []
            joints_path = os.path.join(ep_dir, "joints.jsonl")
            if not os.path.isfile(joints_path):
                return None

            # Performance: checking file existence with os.path.isfile per frame is very slow
            # on networked filesystems (e.g., Colab Drive). Cache directory listing once.
            ep_files: Optional[set[str]] = None
            try:
                ep_files = set(os.listdir(ep_dir))
            except Exception:
                ep_files = None

            parse_warn_action6 = False
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
                    img_path = os.path.join(ep_dir, img_name)
                    if ep_files is not None:
                        if img_name not in ep_files:
                            continue
                    else:
                        if not os.path.isfile(img_path):
                            continue
                    action6_raw = rec.get("action6")
                    action6 = None
                    if isinstance(action6_raw, list):
                        try:
                            action6 = [float(x) for x in action6_raw]
                        except Exception:
                            action6 = None
                            parse_warn_action6 = True
                    elif action6_raw is not None:
                        parse_warn_action6 = True
                    joints_delta = rec.get("joints_delta")
                    records.append(
                        {
                            "image": img_path,
                            "joints": joints,
                            "t": rec.get("t", None),
                            "action6": action6,
                            "joints_delta": joints_delta if isinstance(joints_delta, dict) else None,
                        }
                    )

            if len(records) < self.seq_len:
                return None

            # Determine joint key order
            first_joints = records[0]["joints"]
            joint_keys = meta_joint_keys or sorted(first_joints.keys())

            action_vecs: List[List[float]] = []
            action_dim: Optional[int] = None
            has_action6 = any(r.get("action6") is not None for r in records[1:])
            if has_action6:
                # Prefer recorded teleop action6 vectors; pad/clip to the recorded dimension (default 6).
                # Align actions with transitions: use action attached to arrival frame.
                seen_length: Optional[int] = None
                for r in records[1:]:
                    vec = r.get("action6") or []
                    lenv = len(vec)
                    if lenv == 0:
                        continue
                    if action_dim is None:
                        action_dim = min(ACTION6_DIM, lenv) if lenv > 0 else ACTION6_DIM
                        seen_length = lenv
                    if seen_length is not None and lenv != seen_length:
                        raise ValueError(f"Inconsistent action6 length in episode {ep_dir}")
                    # NOTE: We pad missing elements in action6 with 0.0 to enforce a fixed action_dim.
                    # If zero is meaningful in your action space, replace this padding strategy.
                    padded = list(vec)[:action_dim] + [0.0] * max(0, action_dim - lenv)
                    action_vecs.append([float(v) for v in padded])
                if action_dim is None:
                    action_dim = ACTION6_DIM
                if len(action_vecs) == 0:
                    return None
                action_source = "action6"
            else:
                action_dim = len(joint_keys)
                action_source = self.action_mode
                for i in range(1, len(records)):
                    if self.action_mode == "delta":
                        jd = records[i].get("joints_delta")
                        if isinstance(jd, dict):
                            vals = [float(jd.get(k, 0.0)) for k in joint_keys]
                        else:
                            prev = records[i - 1]["joints"]
                            curr = records[i]["joints"]
                            vals = [float(curr.get(k, 0.0)) - float(prev.get(k, 0.0)) for k in joint_keys]
                    elif self.action_mode == "pos":
                        prev = records[i - 1]["joints"]
                        vals = [float(prev.get(k, 0.0)) for k in joint_keys]
                    else:
                        raise ValueError("action_mode must be 'delta' or 'pos'")
                    action_vecs.append(vals)

            if action_dim is None:
                action_dim = ACTION6_DIM if has_action6 else len(joint_keys)

            actions = torch.tensor(action_vecs, dtype=torch.float32) if action_vecs else torch.zeros((0, action_dim))
            image_paths = [r["image"] for r in records]

            if actions.shape[0] != (len(image_paths) - 1):
                return None

            if parse_warn_action6:
                print(f"[ImageJointSequenceDataset] Warning: some action6 entries could not be parsed in {ep_dir}")

            return {
                "dir": ep_dir,
                "joint_keys": joint_keys,
                "action_dim": int(action_dim),
                "actions": actions,
                "image_paths": image_paths,
                "records": records,
                "action_source": action_source,
            }

        episode_dirs = _discover_episode_dirs(root_dir)
        if not episode_dirs:
            raise FileNotFoundError(f"joints.jsonl not found in {root_dir} or any immediate subdirectory")

        self._episodes: List[Dict] = []
        self.joint_keys: List[str] = []
        self.action_dim: int = 0
        self.num_episodes: int = 0
        action_sum = None
        action_sumsq = None
        total_actions = 0

        dtype = torch.float16 if self._preload_dtype.lower() in ("float16", "fp16") else torch.float32
        self._preloaded: Optional[List[torch.Tensor]] = None
        if self._preload_images:
            self._preloaded = []

        ep_iter = episode_dirs
        if tqdm is not None:
            ep_iter = tqdm(episode_dirs, desc="Loading episodes", unit="ep")

        for ep_dir in ep_iter:
            ep = _load_episode(ep_dir)
            if ep is None:
                continue
            if self.action_dim == 0:
                self.action_dim = int(ep["action_dim"])
                action_sum = torch.zeros(self.action_dim, dtype=torch.float32)
                action_sumsq = torch.zeros(self.action_dim, dtype=torch.float32)
            elif self.action_dim != int(ep["action_dim"]):
                raise ValueError(
                    f"Episode {ep_dir} has action_dim={ep['action_dim']} but first episode uses {self.action_dim}; "
                    "mixing episodes with different action encodings (e.g., action6 vs joint-based) is not supported."
                )

            if not self.joint_keys:
                self.joint_keys = list(ep["joint_keys"])
            elif set(ep["joint_keys"]) != set(self.joint_keys):
                raise ValueError(f"Episode {ep_dir} joint keys differ from first episode")
            self._episodes.append(ep)
            self.num_episodes += 1

            if ep["actions"].numel() > 0:
                total_actions += ep["actions"].shape[0]
                action_sum += ep["actions"].sum(dim=0)
                action_sumsq += (ep["actions"] ** 2).sum(dim=0)

            if self._preloaded is not None:
                imgs_all: List[torch.Tensor] = []
                for p in ep["image_paths"]:
                    img = Image.open(p).convert("RGB")
                    imgs_all.append(self.transform(img).to(dtype=dtype))
                self._preloaded.append(torch.stack(imgs_all, dim=0))

        if not self._episodes:
            raise ValueError(f"No valid episodes with at least seq_len={self.seq_len} found in {root_dir}")

        if self._normalize_actions and total_actions > 0:
            self._action_mean = action_sum / float(total_actions)
            var = (action_sumsq / float(total_actions)) - (self._action_mean ** 2)
            self._action_std = torch.sqrt(var.clamp_min(eps))
        else:
            self._action_mean = torch.zeros(self.action_dim, dtype=torch.float32)
            self._action_std = torch.ones(self.action_dim, dtype=torch.float32)

        # Build mapping of dataset index -> (episode_idx, start_offset)
        self._windows: List[Tuple[int, int]] = []
        for epi, ep in enumerate(self._episodes):
            max_start = len(ep["image_paths"]) - self.seq_len + 1
            for start in range(max_start):
                self._windows.append((epi, start))
        if len(self._windows) == 0:
            raise ValueError(f"No valid sequences of length {self.seq_len} found in dataset {root_dir}")


    def _get_image(self, path: str) -> torch.Tensor:
        if self._preloaded is not None:
            raise RuntimeError("Cannot call _get_image when preload_images is enabled")
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
        return len(self._windows)

    @property
    def windows(self) -> List[Tuple[int, int]]:
        return list(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        epi, start = self._windows[int(idx)]
        end = start + self.seq_len
        ep = self._episodes[epi]

        if self._preloaded is not None:
            images = self._preloaded[epi][start:end]
            # ensure float32 downstream unless caller wants fp16
            if images.dtype != torch.float32:
                images = images.float()
        else:
            imgs = []
            for p in ep["image_paths"][start:end]:
                imgs.append(self._get_image(p))
            images = torch.stack(imgs, dim=0)  # (T,C,H,W)

        actions = ep["actions"][start : end - 1]
        if self._normalize_actions and actions.numel() > 0:
            actions = (actions - self._action_mean) / self._action_std
        return images, actions
