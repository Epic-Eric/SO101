import os
import random
import json
import time
import argparse
from typing import List, Tuple

import torch
from torchvision.io import read_image
import torchvision.transforms.functional as TF

from model.src.models.world_model import WorldModel
from model.src.interfaces.dataset import ImageJointSequenceDataset
from model.src.utils.normalization import get_default_normalization, denormalize


def _tensor_stats(x: torch.Tensor) -> str:
	# Safe stats for debugging (handles empty / non-finite)
	if x.numel() == 0:
		return "empty"
	xf = x.detach()
	finite = torch.isfinite(xf)
	if finite.any():
		xf = xf[finite]
		return f"min={xf.min().item():.3g} max={xf.max().item():.3g} mean={xf.mean().item():.3g} std={xf.std().item():.3g}"
	return "non-finite"


def _zero_fraction(x: torch.Tensor) -> float:
	with torch.no_grad():
		x = x.detach()
		if x.numel() == 0:
			return 0.0
		return float((x == 0).float().mean().item())


def _ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def _save_tensor_image(path: str, img_chw: torch.Tensor, viz_mode: str, norm_params) -> None:
	"""Save a CHW tensor as a PNG after applying visualization mapping."""
	from torchvision.utils import save_image

	x = _viz_to_01(img_chw, viz_mode=viz_mode, norm_params=norm_params)
	save_image(x, path)


def _viz_to_01(img_chw: torch.Tensor, viz_mode: str, norm_params) -> torch.Tensor:
	"""Map a model/image tensor to [0,1] for visualization.

	Modes:
	- "denorm": treat as normalized and apply mean/std via denormalize()
	- "01": treat as already in [0,1]
	- "tanh": treat as [-1,1] and map to [0,1]
	"""
	img_chw = img_chw.float()
	if viz_mode == "01":
		x = img_chw
	elif viz_mode == "tanh":
		x = (img_chw + 1.0) / 2.0
	elif viz_mode == "denorm":
		x = denormalize(img_chw, norm_params)
	else:
		raise ValueError(f"Unknown viz_mode: {viz_mode}")
	return x.clamp(0, 1)


def _guess_viz_mode(img_chw: torch.Tensor) -> str:
	"""Heuristic: choose a visualization mapping based on tensor range."""
	with torch.no_grad():
		x = img_chw.detach().float()
		finite = torch.isfinite(x)
		if not finite.any():
			return "tanh"
		x = x[finite]
		mn = x.min().item()
		mx = x.max().item()
		# Common cases
		if mn >= 0.0 and mx <= 1.0:
			return "01"
		if mn >= -1.05 and mx <= 1.05:
			return "tanh"
		return "denorm"


def _auto_device() -> torch.device:
	if torch.backends.mps.is_available():
		return torch.device("mps")
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def _normalize_image(img: torch.Tensor, norm_params) -> torch.Tensor:
	"""Normalize image tensor with either dict or NormalizationParams.

	Supports shapes (C,H,W) or (T,C,H,W). Values assumed uint8 0..255.
	"""
	img = img.float() / 255.0

	# Extract mean/std from dict-like or attribute-based objects
	if isinstance(norm_params, dict):
		mean_vals = norm_params.get("mean")
		std_vals = norm_params.get("std")
	else:
		mean_vals = getattr(norm_params, "mean", None)
		std_vals = getattr(norm_params, "std", None)

	if mean_vals is None or std_vals is None:
		raise ValueError("Normalization params must provide mean and std")

	mean = torch.tensor(mean_vals, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
	std = torch.tensor(std_vals, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)

	# Broadcast over optional time dimension
	if img.dim() == 3:
		mean = mean.squeeze(0)
		std = std.squeeze(0)

	return (img - mean) / std


def _load_manifest(artifact_dir: str) -> dict:
	manifest_path = os.path.join(artifact_dir, "manifest.json")
	if os.path.exists(manifest_path):
		try:
			with open(manifest_path, "r") as f:
				return json.load(f)
		except Exception:
			return {}
	return {}


def _is_artifact_run_dir(path: str) -> bool:
	return os.path.isdir(path) and os.path.isfile(os.path.join(path, "manifest.json"))


def _pick_latest_run(artifact_base_dir: str) -> str:
	"""Pick the most recent run directory under output/artifacts.

	We consider a "run dir" one that contains a manifest.json.
	"""
	if _is_artifact_run_dir(artifact_base_dir):
		return artifact_base_dir

	if not os.path.isdir(artifact_base_dir):
		raise FileNotFoundError(f"Artifact dir not found: {artifact_base_dir}")

	cands: List[str] = []
	for name in os.listdir(artifact_base_dir):
		p = os.path.join(artifact_base_dir, name)
		if _is_artifact_run_dir(p):
			cands.append(p)
	if not cands:
		raise FileNotFoundError(f"No run dirs with manifest.json under: {artifact_base_dir}")
	# Sort by mtime (newest last)
	cands.sort(key=lambda p: os.path.getmtime(p))
	return cands[-1]


def _resolve_checkpoint(run_dir: str, ckpt_override: str | None) -> str:
	"""Return a checkpoint/model path to load from a run directory."""
	if ckpt_override:
		return ckpt_override
	for name in ("checkpoint_best.pt", "checkpoint_latest.pt", "model_final.pt"):
		p = os.path.join(run_dir, name)
		if os.path.isfile(p):
			return p
	raise FileNotFoundError(f"No checkpoint found in {run_dir} (expected checkpoint_best.pt / checkpoint_latest.pt / model_final.pt)")


def _load_checkpoint_state(ckpt_path: str) -> dict:
	obj = torch.load(ckpt_path, map_location="cpu")
	if isinstance(obj, dict):
		if "model_state" in obj:
			return obj["model_state"]
		if "state_dict" in obj:
			return obj["state_dict"]
		# some checkpoints might store the state dict at top-level
		return obj
	raise RuntimeError("Unsupported checkpoint format")



def _discover_episode_dirs(root: str) -> List[str]:
	"""Discover episode dirs containing joints.jsonl (root itself or immediate children)."""
	if os.path.isfile(os.path.join(root, "joints.jsonl")):
		return [root]
	try:
		immediate: List[str] = []
		for name in sorted(os.listdir(root)):
			ep = os.path.join(root, name)
			if os.path.isdir(ep) and os.path.isfile(os.path.join(ep, "joints.jsonl")):
				immediate.append(ep)
		if immediate:
			return immediate
	except Exception:
		pass
	# Fallback to recursive walk (slower)
	eps: List[str] = []
	for dirpath, _, filenames in os.walk(root):
		if "joints.jsonl" in filenames:
			eps.append(dirpath)
	return sorted(eps)


def _pick_episode_dir(data_root: str, episode: str | None) -> str:
	"""Pick an episode directory.

	- If episode is a path to a dir containing joints.jsonl, use it.
	- If episode is a name, resolve under data_root.
	- Otherwise, pick a random episode under data_root.
	"""
	if episode:
		if os.path.isdir(episode) and os.path.isfile(os.path.join(episode, "joints.jsonl")):
			return episode
		cand = os.path.join(data_root, episode)
		if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "joints.jsonl")):
			return cand
		raise FileNotFoundError(f"Episode not found or missing joints.jsonl: {episode}")

	eps = _discover_episode_dirs(data_root)
	if not eps:
		raise FileNotFoundError(f"No episodes found under: {data_root}")
	return random.choice(eps)


def _load_episode_records(ep_dir: str) -> List[dict]:
	joints_path = os.path.join(ep_dir, "joints.jsonl")
	if not os.path.exists(joints_path):
		raise FileNotFoundError(f"Missing joints file: {joints_path}")

	records: List[dict] = []
	with open(joints_path, "r") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception:
				continue
			img = obj.get("image")
			j = obj.get("joints")
			if not isinstance(img, str) or not isinstance(j, dict):
				continue
			records.append(obj)
	return records


def _list_sequence(data_root: str, seq_len: int, episode: str | None) -> Tuple[str, List[str], List[dict]]:
	ep_dir = _pick_episode_dir(data_root, episode)
	records = _load_episode_records(ep_dir)
	# Only keep entries that have an existing image file
	filtered: List[dict] = []
	try:
		ep_files = set(os.listdir(ep_dir))
	except Exception:
		ep_files = None
	for r in records:
		name = r.get("image")
		if not isinstance(name, str):
			continue
		if ep_files is not None:
			if name not in ep_files:
				continue
		else:
			if not os.path.isfile(os.path.join(ep_dir, name)):
				continue
		filtered.append(r)

	if len(filtered) < seq_len:
		raise ValueError(f"Not enough frames in episode {ep_dir} for sequence length {seq_len} (have {len(filtered)})")

	start = random.randint(0, len(filtered) - seq_len)
	seq = filtered[start : start + seq_len]
	frame_paths = [os.path.join(ep_dir, r["image"]) for r in seq]
	return ep_dir, frame_paths, seq


def _compute_actions_from_records(records: List[dict], action_mode: str, joint_keys: List[str]) -> torch.Tensor:
	"""Match training dataset behavior.

	- Prefer recorded action6 vectors when present.
	- Otherwise use joints_delta if available (delta mode) or compute deltas from joints.
	"""
	T = len(records)
	if T < 2:
		raise ValueError("Need at least 2 records to build actions")

	# Prefer action6 if available on most records
	has_action6 = any(isinstance(r.get("action6"), list) for r in records[1:])
	if has_action6:
		vecs: List[List[float]] = []
		# Align actions with transitions: use action attached to arrival frame.
		for r in records[1:]:
			v = r.get("action6")
			if not isinstance(v, list):
				v = []
			vecs.append([float(x) for x in v])
		# Pad/clip to fixed dim
		A = max(1, min(6, max((len(v) for v in vecs), default=6)))
		out = torch.zeros(T - 1, A)
		for i, v in enumerate(vecs):
			for j in range(min(A, len(v))):
				out[i, j] = float(v[j])
		return out

	# Else derive from joints
	A = len(joint_keys)
	out = torch.zeros(T - 1, A)
	for i in range(1, T):
		prev = records[i - 1].get("joints") or {}
		curr = records[i].get("joints") or {}
		if action_mode == "delta":
			jd = records[i].get("joints_delta")
			if isinstance(jd, dict):
				vals = [float(jd.get(k, 0.0)) for k in joint_keys]
			else:
				vals = [float(curr.get(k, 0.0)) - float(prev.get(k, 0.0)) for k in joint_keys]
		elif action_mode == "pos":
			vals = [float(prev.get(k, 0.0)) for k in joint_keys]
		else:
			raise ValueError("action_mode must be 'delta' or 'pos'")
		out[i - 1] = torch.tensor(vals, dtype=out.dtype)
	return out


def _compute_delta_actions(joints_seq: List[dict], order: List[str]) -> torch.Tensor:
	T = len(joints_seq)
	A = len(order)
	actions = torch.zeros(T - 1, A)
	for i in range(1, T):
		prev = joints_seq[i - 1]
		curr = joints_seq[i]
		for j, k in enumerate(order):
			actions[i - 1, j] = float(curr.get(k, 0.0)) - float(prev.get(k, 0.0))
	return actions


def _to_surface(img: torch.Tensor, norm_params, scale: int = 6):
	import pygame

	# Default visualization assumes model outputs normalized (mean/std) space.
	x = _viz_to_01(img, viz_mode="denorm", norm_params=norm_params)
	arr = (x * 255.0).byte().permute(1, 2, 0).contiguous().cpu().numpy()  # (H,W,3)
	h, w, _ = arr.shape
	# Use fromstring (copy) to avoid referencing a temporary buffer.
	surf = pygame.image.fromstring(arr.tobytes(), (w, h), "RGB")
	# convert() requires an initialized display mode
	if pygame.display.get_init() and pygame.display.get_surface() is not None:
		surf = surf.convert()
	w, h = surf.get_size()
	return pygame.transform.scale(surf, (w * scale, h * scale))


def _to_surface_viz(img: torch.Tensor, norm_params, viz_mode: str, scale: int = 6):
	import pygame

	x = _viz_to_01(img, viz_mode=viz_mode, norm_params=norm_params)
	arr = (x * 255.0).byte().permute(1, 2, 0).contiguous().cpu().numpy()  # (H,W,3)
	h, w, _ = arr.shape
	# Use fromstring (copy) to avoid referencing a temporary buffer.
	surf = pygame.image.fromstring(arr.tobytes(), (w, h), "RGB")
	# convert() requires an initialized display mode
	if pygame.display.get_init() and pygame.display.get_surface() is not None:
		surf = surf.convert()
	w, h = surf.get_size()
	return pygame.transform.scale(surf, (w * scale, h * scale))


def _prepare_image(img: torch.Tensor, target_size: int = 64) -> torch.Tensor:
	"""Center-crop to square and resize to target_size.

	Expects a uint8 tensor (C,H,W).
	"""
	c, h, w = img.shape
	# Center-crop to square
	side = min(h, w)
	img = TF.center_crop(img, [side, side])
	# Resize to target size
	img = TF.resize(img, [target_size, target_size], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
	return img


def main():
	parser = argparse.ArgumentParser(description="Interactive world-model inference viewer")
	parser.add_argument("--data-root", default=os.path.join("data", "world_model_episodes"), help="Episode root dir (contains episode_*/joints.jsonl)")
	parser.add_argument(
		"--artifact-dir",
		default=os.path.join("output", "artifacts"),
		help="Either a run dir (contains manifest.json) or the base output/artifacts dir (auto-picks latest run)",
	)
	parser.add_argument("--ckpt", default=None, help="Override checkpoint path")
	parser.add_argument("--episode", default=None, help="Episode folder name under data-root (or full path to episode dir)")
	parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length")
	parser.add_argument("--image-size", type=int, default=None, help="Override image size")
	parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging and overlay")
	parser.add_argument("--headless", action="store_true", help="No pygame window; dump debug images and exit")
	parser.add_argument("--dump-dir", default=os.path.join("output", "debug_frames"), help="Where to save debug frames")
	parser.add_argument(
		"--headless-steps",
		type=int,
		default=8,
		help="When --headless is set, how many RSSM prediction steps to dump",
	)
	parser.add_argument(
		"--max-frames",
		type=int,
		default=0,
		help="If >0, exit after this many GUI frames (useful with SDL_VIDEODRIVER=dummy)",
	)
	parser.add_argument(
		"--stochastic",
		action="store_true",
		help="Use stochastic RSSM sampling (default is deterministic prior mean)",
	)
	parser.add_argument(
		"--prior-temperature",
		type=float,
		default=1.0,
		help="When sampling (i.e. --stochastic), scale the RSSM prior std by this factor. Larger values can reduce fixed-point collapse in long rollouts.",
	)
	parser.add_argument(
		"--idle-action-noise",
		type=float,
		default=0.0,
		help="If --idle-step is set and no keys are pressed, add N(0, this^2) noise to actions (in --action-input units). Can help avoid converging to a single latent.",
	)
	parser.add_argument(
		"--half",
		action="store_true",
		help="Run model in qfp16 on CUDA (can be unstable; default off)",
	)
	parser.add_argument(
		"--action-input",
		choices=["normalized", "raw"],
		default="normalized",
		help="Keyboard action space. 'normalized' matches training (mean=0,std=1). 'raw' will be normalized using dataset stats.",
	)
	parser.add_argument(
		"--action-step",
		type=float,
		default=0.01,
		help="Per-keypress action magnitude (in units of --action-input)",
	)
	parser.add_argument(
		"--idle-step",
		action="store_true",
		help="Advance RSSM even when no action is pressed (default: hold state)",
	)
	parser.add_argument(
		"--viz",
		choices=["auto", "01", "tanh", "denorm"],
		default="auto",
		help="Initial visualization mapping. 'auto' guesses from x_rec0 range.",
	)
	args = parser.parse_args()
	if args.prior_temperature <= 0:
		raise ValueError("--prior-temperature must be > 0")
	if args.idle_action_noise < 0:
		raise ValueError("--idle-action-noise must be >= 0")

	data_root = os.path.join("data", "world_model_episodes")
	artifact_dir = os.path.join("output", "artifacts")
	if args.data_root:
		data_root = args.data_root
	if args.artifact_dir:
		artifact_dir = args.artifact_dir

	run_dir = _pick_latest_run(artifact_dir)
	ckpt_path = _resolve_checkpoint(run_dir, args.ckpt)

	dev = _auto_device()
	norm_params = get_default_normalization()

	manifest = _load_manifest(run_dir)
	latent_dim = int(manifest.get("latent_dim", 128))
	deter_dim = int(manifest.get("deter_dim", 256))
	seq_len = int(manifest.get("seq_len", 16))
	base_channels = int(manifest.get("base_channels", 64))
	rec_loss = manifest.get("rec_loss", "mse")
	output_activation = manifest.get("output_activation", "tanh")
	kl_beta = float(manifest.get("kl_beta", 1.0))
	free_nats = float(manifest.get("free_nats", 0.0))
	image_size = int(manifest.get("image_size", 64))
	action_mode = str(manifest.get("action_mode", "delta"))
	manifest_joint_keys = manifest.get("joint_keys")
	joint_keys: List[str] = [str(x) for x in manifest_joint_keys] if isinstance(manifest_joint_keys, list) else []
	manifest_action_dim = manifest.get("action_dim")
	if args.seq_len is not None:
		seq_len = int(args.seq_len)
	if args.image_size is not None:
		image_size = int(args.image_size)

	if args.debug:
		print(f"[debug] device={dev}")
		print(f"[debug] data_root={data_root}")
		print(f"[debug] artifact_dir={artifact_dir}")
		print(f"[debug] run_dir={run_dir}")
		print(f"[debug] ckpt_path={ckpt_path}")
		print(f"[debug] manifest keys={sorted(list(manifest.keys()))}")
		print(
			"[debug] model cfg "
			f"latent_dim={latent_dim} deter_dim={deter_dim} base_channels={base_channels} "
			f"rec_loss={rec_loss} output_activation={output_activation} kl_beta={kl_beta} free_nats={free_nats} "
			f"seq_len={seq_len} image_size={image_size}"
		)

	# Initial guess; will be refined after we compute x_rec0.
	if str(output_activation).lower() == "tanh":
		default_viz_mode = "tanh"
	elif str(output_activation).lower() in {"sigmoid", "01", "zero_one"}:
		default_viz_mode = "01"
	else:
		default_viz_mode = "denorm"

	ep_dir, frame_paths, records_seq = _list_sequence(data_root, seq_len, args.episode)
	if not joint_keys:
		# Fallback: infer joint order from first record
		j0 = records_seq[0].get("joints")
		joint_keys = sorted(list(j0.keys())) if isinstance(j0, dict) else []
	actions_seq = _compute_actions_from_records(records_seq, action_mode=action_mode, joint_keys=joint_keys)
	action_dim = int(manifest_action_dim) if isinstance(manifest_action_dim, (int, float)) else int(actions_seq.shape[1])

	# Training normalizes actions using dataset-wide mean/std. If we want to drive the model
	# with raw actions, we must apply the same normalization here.
	action_mean = torch.zeros(action_dim, dtype=torch.float32)
	action_std = torch.ones(action_dim, dtype=torch.float32)
	if args.action_input == "raw":
		try:
			stats_ds = ImageJointSequenceDataset(
				root_dir=data_root,
				seq_len=2,
				image_size=image_size,
				normalize_images=False,
				norm_params=norm_params,
				action_mode=action_mode,
				normalize_actions=True,
				cache_images=False,
				preload_images=False,
			)
			am = getattr(stats_ds, "_action_mean", None)
			as_ = getattr(stats_ds, "_action_std", None)
			if isinstance(am, torch.Tensor) and isinstance(as_, torch.Tensor):
				action_mean = am.float().cpu()
				action_std = as_.float().cpu()
			if action_mean.numel() != action_dim or action_std.numel() != action_dim:
				raise RuntimeError(
					f"Action stats dim mismatch: stats {action_mean.numel()} vs action_dim {action_dim}. "
					"This can happen if episodes mix action encodings (action6 vs joint-delta)."
				)
		except Exception as e:
			print(f"[warn] could not compute action mean/std; raw actions may have weak/no effect: {e}")
	if args.debug:
		print(f"[debug] episode_dir={ep_dir}")
		print(f"[debug] action_mode={action_mode} joint_keys_dim={len(joint_keys)} action_dim={action_dim}")
		print(f"[debug] action_input={args.action_input} action_step={float(args.action_step)}")
		if args.action_input == "raw":
			print(f"[debug] action_mean: {action_mean.numpy().round(4).tolist()}")
			print(f"[debug] action_std:  {action_std.numpy().round(4).tolist()}")

	model = WorldModel(
		action_dim=action_dim,
		latent_dim=latent_dim,
		deter_dim=deter_dim,
		base_channels=base_channels,
		rec_loss=rec_loss,
		output_activation=output_activation,
		kl_beta=kl_beta,
		free_nats=free_nats,
	).to(dev)

	if not os.path.exists(ckpt_path):
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
	state = _load_checkpoint_state(ckpt_path)
	missing, unexpected = model.load_state_dict(state, strict=False)
	if args.debug:
		print(f"[debug] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")
		if len(missing) > 0:
			print("[debug] missing keys (first 20):", missing[:20])
		if len(unexpected) > 0:
			print("[debug] unexpected keys (first 20):", unexpected[:20])
	model.eval()

	def _to_model_action(action_vec: torch.Tensor) -> torch.Tensor:
		"""Convert an (1,A) action into the model's expected (normalized) action space.

		Training uses dataset-normalized actions, so for interactive inference we either:
		- treat keyboard actions as already normalized (default), or
		- normalize raw actions using dataset mean/std.
		
		This adapts to the model's current dtype (e.g., after model.half()).
		"""
		dtype_now = next(model.parameters()).dtype
		a = action_vec.to(device=dev, dtype=dtype_now)
		if args.action_input == "raw":
			mean_now = action_mean.to(device=dev, dtype=dtype_now).view(1, -1)
			std_now = action_std.to(device=dev, dtype=dtype_now).view(1, -1)
			return (a - mean_now) / std_now
		return a

	def _sample_prior(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		"""Sample z ~ N(mu, (temperature*std)^2) for inference-time rollouts."""
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std * float(args.prior_temperature)

	imgs = []
	for p in frame_paths:
		img = read_image(p)  # (C,H,W) uint8
		# Preprocess to expected model size
		img = _prepare_image(img, target_size=image_size)
		imgs.append(img)
		if args.debug and len(imgs) == 1:
			print(f"[debug] first raw image path={p}")
			print(f"[debug] first raw image uint8 stats: {img.dtype} shape={tuple(img.shape)}")
	# Keep images on CPU to reduce GPU memory; we only need the first frame
	images = torch.stack(imgs, dim=0)  # (T,C,H,W) CPU
	if args.debug:
		# Show raw range pre-normalization
		x_raw = (images.float() / 255.0)
		print(f"[debug] raw images in [0,1] stats: {_tensor_stats(x_raw)}")
	images = _normalize_image(images, norm_params)
	if args.debug:
		print(f"[debug] normalized images stats: {_tensor_stats(images)}")
	start_image = images[0].unsqueeze(0).to(dev)  # (1,C,H,W)

	# Debug dump of the start frame (lets us confirm data & mapping even without pygame)
	if args.debug or args.headless:
		_ensure_dir(args.dump_dir)
		start_raw = imgs[0].float() / 255.0
		_save_tensor_image(os.path.join(args.dump_dir, "start_raw_01.png"), start_raw, viz_mode="01", norm_params=norm_params)
		_save_tensor_image(os.path.join(args.dump_dir, "start_norm_denorm.png"), images[0].cpu(), viz_mode="denorm", norm_params=norm_params)
		_save_tensor_image(os.path.join(args.dump_dir, "start_norm_01.png"), images[0].cpu(), viz_mode="01", norm_params=norm_params)
		_save_tensor_image(os.path.join(args.dump_dir, "start_norm_tanh.png"), images[0].cpu(), viz_mode="tanh", norm_params=norm_params)

	if args.headless:
		# Run one encode/decode and dump reconstructions to isolate normalization/range issues.
		with torch.no_grad():
			x_rec0, mu0, logvar0, _ = model.vae(start_image)
		if args.debug:
			print(f"[debug] x_rec0 stats: {_tensor_stats(x_rec0)}")
			print(f"[debug] mu0 stats: {_tensor_stats(mu0)}")
			print(f"[debug] logvar0 stats: {_tensor_stats(logvar0)}")
		_save_tensor_image(os.path.join(args.dump_dir, "x_rec0_denorm.png"), x_rec0[0].cpu(), viz_mode="denorm", norm_params=norm_params)
		_save_tensor_image(os.path.join(args.dump_dir, "x_rec0_01.png"), x_rec0[0].cpu(), viz_mode="01", norm_params=norm_params)
		_save_tensor_image(os.path.join(args.dump_dir, "x_rec0_tanh.png"), x_rec0[0].cpu(), viz_mode="tanh", norm_params=norm_params)

		# Also dump a few RSSM predicted frames (this is the common source of "black screen" reports).
		steps = max(0, int(args.headless_steps))
		if steps > 0:
			with torch.no_grad():
				z = model.vae.reparameterize(mu0, logvar0)
				state_rssm = model.rssm.init_state(batch_size=1, device=dev)
				# Important: initialize h using the start latent so the prior isn't effectively unconditioned.
				model_dtype = next(model.parameters()).dtype
				zero_action = torch.zeros(1, action_dim, device=dev, dtype=model_dtype)
				z0 = z.to(model_dtype)
				x0 = model.rssm.inp(torch.cat([z0, _to_model_action(zero_action)], dim=-1))
				h0 = model.rssm.gru(x0, state_rssm.h.to(model_dtype))
				state_rssm = type(state_rssm)(h=h0, z=z0)
				for i in range(steps):
					state_next, (mu_p, logvar_p) = model.rssm.step(state_rssm, _to_model_action(zero_action))
					if args.stochastic:
						z_next = _sample_prior(mu_p, logvar_p)
						state_rssm = type(state_next)(h=state_next.h, z=z_next)
					else:
						state_rssm = type(state_next)(h=state_next.h, z=mu_p)
					pred = model.vae.decode(state_rssm.z)[0]
					if args.debug and i in (0, steps - 1):
						print(f"[debug] pred step {i} stats: {_tensor_stats(pred)}")
					for mode in ("denorm", "01", "tanh"):
						out_path = os.path.join(args.dump_dir, f"wm_step_{i:03d}_{mode}.png")
						_save_tensor_image(out_path, pred.detach().cpu(), viz_mode=mode, norm_params=norm_params)
		print(f"[headless] wrote debug images to {args.dump_dir}")
		return

	try:
		import pygame
	except ModuleNotFoundError as e:
		raise ModuleNotFoundError(
			"pygame is required for interactive mode. Install it (e.g. `pip install pygame`) "
			"or run with `--headless` to dump debug images without a window."
		) from e

	pygame.init()
	pygame.display.set_caption("World Model Inference")
	# Initial reconstruction from VAE only (avoid full-sequence forward)
	with torch.no_grad():
		x_rec0, mu0, logvar0, _ = model.vae(start_image)
	# Refine visualization choice using actual tensor range (helps when manifest is missing/wrong).
	if args.viz == "auto":
		guessed = _guess_viz_mode(x_rec0[0])
		default_viz_mode = guessed
	else:
		default_viz_mode = args.viz
	if args.debug:
		print(f"[debug] x_rec0 range: {_tensor_stats(x_rec0[0])}")
		print(f"[debug] viz requested={args.viz} default_viz_mode={default_viz_mode}")
		print(f"[debug] initial x_rec0 stats: {_tensor_stats(x_rec0)}")
		print(f"[debug] initial x_rec0[0] stats: {_tensor_stats(x_rec0[0])}")
		print(f"[debug] initial x_rec0 denorm stats: {_tensor_stats(_viz_to_01(x_rec0[0], 'denorm', norm_params))}")
	surface0 = _to_surface_viz(x_rec0[0], norm_params, viz_mode=default_viz_mode)
	screen = pygame.display.set_mode(surface0.get_size())

	# Debug: allow showing the raw JPEG directly via pygame's loader (bypasses tensor->surface path)
	jpg_surface = None
	try:
		jpg_path = frame_paths[0]
		jpg_surface = pygame.image.load(jpg_path).convert()
		jpg_surface = pygame.transform.scale(jpg_surface, screen.get_size())
		if args.debug:
			print(f"[debug] loaded jpg via pygame: {jpg_path}")
	except Exception as e:
		if args.debug:
			print(f"[debug] failed to load jpg via pygame: {e}")

	# Fonts can fail in minimal environments; keep overlay optional
	try:
		font = pygame.font.SysFont(None, 18)
	except Exception:
		font = None

	keymap_pos = {
		pygame.K_q: 0,
		pygame.K_w: 1,
		pygame.K_e: 2,
		pygame.K_r: 3,
		pygame.K_t: 4,
		pygame.K_y: 5,
	}
	keymap_neg = {
		pygame.K_a: 0,
		pygame.K_s: 1,
		pygame.K_d: 2,
		pygame.K_f: 3,
		pygame.K_g: 4,
		pygame.K_h: 5,
	}

	# One-time control map print (helps when actions correspond to joints).
	key_names = {
		pygame.K_q: "q",
		pygame.K_w: "w",
		pygame.K_e: "e",
		pygame.K_r: "r",
		pygame.K_t: "t",
		pygame.K_y: "y",
		pygame.K_a: "a",
		pygame.K_s: "s",
		pygame.K_d: "d",
		pygame.K_f: "f",
		pygame.K_g: "g",
		pygame.K_h: "h",
	}
	labels: List[str] = []
	if joint_keys and len(joint_keys) == action_dim:
		labels = [str(k) for k in joint_keys]
	else:
		labels = [f"action[{i}]" for i in range(action_dim)]

	step_size = float(args.action_step)

	print("Controls:")
	print("  Hold keys for continuous action")
	print("  q w e r t y = + action on dims 0..5")
	print("  a s d f g h = - action on dims 0..5")
	print("  (optional) Shift reverses the sign")
	print(f"  Action input space: {args.action_input} (step={step_size})")
	print("  Key mapping (only first 6 action dims are bound):")
	for idx in range(min(6, action_dim)):
		lab = labels[idx] if 0 <= idx < len(labels) else f"action[{idx}]"
		pos_key = None
		neg_key = None
		for k, i in keymap_pos.items():
			if i == idx:
				pos_key = key_names.get(k, str(k))
		for k, i in keymap_neg.items():
			if i == idx:
				neg_key = key_names.get(k, str(k))
		print(f"    dim {idx}: {lab}   (+){pos_key}  (-){neg_key}")

	current_img = x_rec0[0].clone()

	# Debug visualization helpers
	viz_modes = [default_viz_mode] + [m for m in ["denorm", "01", "tanh"] if m != default_viz_mode]
	viz_idx = 0
	show_overlay = bool(args.debug)
	show_start_raw = False
	show_jpg = False
	stochastic = bool(args.stochastic)
	frame_idx = 0
	last_dump_t = 0.0

	running = True
	clock = pygame.time.Clock()

	# Optimize GPU memory: use half precision and autocast when on CUDA
	use_cuda = (dev.type == "cuda")
	use_half = bool(args.half) and use_cuda
	if use_half:
		model.half()
		start_image = start_image.half()
	model_dtype = next(model.parameters()).dtype

	with torch.no_grad():
		x_rec0, mu0, logvar0, _ = model.vae(start_image)
		z = model.vae.reparameterize(mu0, logvar0)
		state_rssm = model.rssm.init_state(batch_size=1, device=dev)
		# Important: initialize h using the start latent so the prior isn't effectively unconditioned.
		zero_action = torch.zeros(1, action_dim, device=dev, dtype=model_dtype)
		z0 = z.to(model_dtype)
		x0 = model.rssm.inp(torch.cat([z0, _to_model_action(zero_action)], dim=-1))
		h0 = model.rssm.gru(x0, state_rssm.h.to(model_dtype))
		state_rssm = type(state_rssm)(h=h0, z=z0)
		current_img = x_rec0[0]

		while running:
			# Base action for this tick comes from currently held keys.
			action = torch.zeros(1, action_dim, device=dev, dtype=model_dtype)
			action_applied = False

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						running = False
					if event.key == pygame.K_TAB:
						show_overlay = not show_overlay
					if event.key == pygame.K_v:
						viz_idx = (viz_idx + 1) % len(viz_modes)
					if event.key == pygame.K_o:
						show_start_raw = not show_start_raw
					if event.key == pygame.K_j:
						show_jpg = not show_jpg
					if event.key == pygame.K_SPACE:
						stochastic = not stochastic
					if event.key == pygame.K_p:
						# Dump current frame (throttle a bit)
						now = time.time()
						if now - last_dump_t > 0.15:
							_ensure_dir(args.dump_dir)
							mode = viz_modes[viz_idx]
							out_path = os.path.join(args.dump_dir, f"frame_{frame_idx:06d}_{mode}.png")
							_save_tensor_image(out_path, current_img.detach().cpu(), viz_mode=mode, norm_params=norm_params)
							print(f"[dump] wrote {out_path} | dtype={current_img.dtype} stats={_tensor_stats(current_img)}")
							last_dump_t = now
					# Action keys are handled via held-key polling below.
					pass

			# Held-key polling for continuous actions
			pressed = pygame.key.get_pressed()
			shift_down = bool(pressed[pygame.K_LSHIFT] or pressed[pygame.K_RSHIFT])
			for k, idx in keymap_pos.items():
				if 0 <= idx < action_dim and pressed[k]:
					sign = -1.0 if shift_down else 1.0
					action[0, idx] += float(sign) * step_size
					action_applied = True
			for k, idx in keymap_neg.items():
				if 0 <= idx < action_dim and pressed[k]:
					sign = 1.0 if shift_down else -1.0
					action[0, idx] += float(sign) * step_size
					action_applied = True

			# Optionally hold the state when there is no input.
			if not args.idle_step and not action_applied:
				pass
			else:
				if args.idle_step and (not action_applied) and float(args.idle_action_noise) > 0:
					action = action + torch.randn_like(action) * float(args.idle_action_noise)
				if use_cuda:
					with torch.cuda.amp.autocast(enabled=use_half):
						state_next, (mu_p, logvar_p) = model.rssm.step(state_rssm, _to_model_action(action))
						if stochastic:
							z_next = _sample_prior(mu_p, logvar_p)
							state_rssm = type(state_next)(h=state_next.h, z=z_next)
						else:
							state_rssm = type(state_next)(h=state_next.h, z=mu_p)
						current_img = model.vae.decode(state_rssm.z)[0]
				else:
					state_next, (mu_p, logvar_p) = model.rssm.step(state_rssm, _to_model_action(action))
					if stochastic:
						z_next = _sample_prior(mu_p, logvar_p)
						state_rssm = type(state_next)(h=state_next.h, z=z_next)
					else:
						state_rssm = type(state_next)(h=state_next.h, z=mu_p)
					current_img = model.vae.decode(state_rssm.z)[0]

			mode = viz_modes[viz_idx]
			if show_jpg and jpg_surface is not None:
				screen.blit(jpg_surface, (0, 0))
			elif show_start_raw:
				# Show the original loaded start image (sanity-check pygame pipeline)
				raw_chw = (imgs[0].float() / 255.0)
				surf = _to_surface_viz(raw_chw, norm_params, viz_mode="01")
				screen.blit(surf, (0, 0))
			else:
				surf = _to_surface_viz(current_img, norm_params, viz_mode=mode)
				screen.blit(surf, (0, 0))

			if show_overlay and font is not None:
				fps = clock.get_fps()
				lines = [
					f"FPS {fps:.1f}  dev={dev.type}  viz={mode}  fp16={use_half}  {'stoch' if stochastic else 'det'}(T={args.prior_temperature:g})  {'idle-step' if args.idle_step else 'hold'}",
					"Keys (hold): qwerty=+ action, asdfgh=- action | Shift flips sign | v=viz | o=orig | j=jpg | Space=stoch | Tab=overlay | p=dump | esc=quit",
					f"img stats: {_tensor_stats(current_img)} dtype={str(current_img.dtype)} z0%={_zero_fraction(current_img)*100:.1f}",
					f"img->01 stats: {_tensor_stats(_viz_to_01(current_img, mode, norm_params))}",
				]
				if action.abs().sum().item() > 0:
					lines.append(f"action: {action.detach().cpu().numpy().round(3).tolist()[0]}")
				for i, t in enumerate(lines):
					s = font.render(t, True, (255, 255, 255))
					screen.blit(s, (6, 6 + 18 * i))

			pygame.display.flip()
			# Optional auto-exit for scripted debugging (e.g. SDL_VIDEODRIVER=dummy)
			if args.max_frames and frame_idx == 0 and (args.debug or args.dump_dir):
				try:
					_ensure_dir(args.dump_dir)
					snap_path = os.path.join(args.dump_dir, f"screen_frame_{frame_idx:06d}.png")
					pygame.image.save(screen, snap_path)
					print(f"[dump] wrote {snap_path}")
				except Exception as e:
					if args.debug:
						print(f"[debug] failed to save screen snapshot: {e}")
			clock.tick(30)
			frame_idx += 1
			if args.max_frames and frame_idx >= args.max_frames:
				running = False

	pygame.quit()


if __name__ == "__main__":
	main()