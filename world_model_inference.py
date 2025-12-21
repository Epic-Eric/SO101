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


def _list_sequence(root: str, seq_len: int) -> Tuple[List[str], List[dict]]:
	joints_path = os.path.join(root, "joints.jsonl")
	if not os.path.exists(joints_path):
		raise FileNotFoundError(f"Missing joints file: {joints_path}")

	frames: List[str] = []
	joints: List[dict] = []
	with open(joints_path, "r") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
				frames.append(obj["image"])
				joints.append(obj["joints"]) 
			except Exception:
				continue

	if len(frames) < seq_len:
		raise ValueError(f"Not enough frames for sequence length {seq_len}")

	start = random.randint(0, len(frames) - seq_len)
	return frames[start : start + seq_len], joints[start : start + seq_len]


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
	parser.add_argument("--data-root", default=os.path.join("data", "captured_images_and_joints"))
	parser.add_argument("--artifact-dir", default=os.path.join("output", "artifacts"))
	parser.add_argument("--ckpt", default=None, help="Override checkpoint path")
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
		"--half",
		action="store_true",
		help="Run model in qfp16 on CUDA (can be unstable; default off)",
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

	data_root = os.path.join("data", "captured_images_and_joints")
	artifact_dir = os.path.join("output", "artifacts")
	if args.data_root:
		data_root = args.data_root
	if args.artifact_dir:
		artifact_dir = args.artifact_dir
	ckpt_path = args.ckpt or os.path.join(artifact_dir, "checkpoint_best.pt")

	dev = _auto_device()
	norm_params = get_default_normalization()

	manifest = _load_manifest(artifact_dir)
	latent_dim = int(manifest.get("latent_dim", 128))
	deter_dim = int(manifest.get("deter_dim", 256))
	seq_len = int(manifest.get("seq_len", 16))
	base_channels = int(manifest.get("base_channels", 64))
	rec_loss = manifest.get("rec_loss", "mse")
	output_activation = manifest.get("output_activation", "tanh")
	kl_beta = float(manifest.get("kl_beta", 1.0))
	free_nats = float(manifest.get("free_nats", 0.0))
	image_size = int(manifest.get("image_size", 64))
	if args.seq_len is not None:
		seq_len = int(args.seq_len)
	if args.image_size is not None:
		image_size = int(args.image_size)

	if args.debug:
		print(f"[debug] device={dev}")
		print(f"[debug] data_root={data_root}")
		print(f"[debug] artifact_dir={artifact_dir}")
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

	action_order = [
		"shoulder_pan.pos",
		"shoulder_lift.pos",
		"elbow_flex.pos",
		"wrist_flex.pos",
		"wrist_roll.pos",
		"gripper.pos",
	]

	frames, joints_seq = _list_sequence(data_root, seq_len)
	actions_seq = _compute_delta_actions(joints_seq, action_order)

	model = WorldModel(
		action_dim=len(action_order),
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

	imgs = []
	for name in frames:
		p = os.path.join(data_root, name)
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
				state_rssm = type(state_rssm)(h=state_rssm.h, z=z)
				model_dtype = next(model.parameters()).dtype
				zero_action = torch.zeros(1, len(action_order), device=dev, dtype=model_dtype)
				for i in range(steps):
					state_rssm, _ = model.rssm.step(state_rssm, zero_action)
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
		jpg_path = os.path.join(data_root, frames[0])
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

	keymap = {
		pygame.K_q: 0,
		pygame.K_w: 1,
		pygame.K_e: 2,
		pygame.K_r: 3,
		pygame.K_t: 4,
		pygame.K_y: 5,
	}

	step_size = 0.2
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

	with torch.no_grad():
		x_rec0, mu0, logvar0, _ = model.vae(start_image)
		z = model.vae.reparameterize(mu0, logvar0)
		state_rssm = model.rssm.init_state(batch_size=1, device=dev)
		# Important: initialize h using the start latent so the prior isn't effectively unconditioned.
		model_dtype = next(model.parameters()).dtype
		zero_action = torch.zeros(1, len(action_order), device=dev, dtype=model_dtype)
		x0 = model.rssm.inp(torch.cat([z.to(model_dtype), zero_action], dim=-1))
		h0 = model.rssm.gru(x0, state_rssm.h.to(model_dtype))
		state_rssm = type(state_rssm)(h=h0, z=z.to(model_dtype))
		current_img = x_rec0[0]

		while running:
			# Match action dtype to model parameters
			action = torch.zeros(1, len(action_order), device=dev, dtype=model_dtype)
			action_applied = False
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						running = False
					if event.key == pygame.K_d:
						show_overlay = not show_overlay
					if event.key == pygame.K_v:
						viz_idx = (viz_idx + 1) % len(viz_modes)
					if event.key == pygame.K_o:
						show_start_raw = not show_start_raw
					if event.key == pygame.K_j:
						show_jpg = not show_jpg
					if event.key == pygame.K_s:
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
					idx = keymap.get(event.key, None)
					if idx is not None:
						val = -step_size if (event.mod & pygame.KMOD_SHIFT) else step_size
						action[0, idx] = val
						action_applied = True

			# Optionally hold the state when there is no input.
			if not args.idle_step and not action_applied:
				pass
			else:
				if use_cuda:
					with torch.cuda.amp.autocast(enabled=use_half):
						state_next, (mu_p, logvar_p) = model.rssm.step(state_rssm, action)
						if stochastic:
							state_rssm = state_next
						else:
							state_rssm = type(state_next)(h=state_next.h, z=mu_p)
						current_img = model.vae.decode(state_rssm.z)[0]
				else:
					state_next, (mu_p, logvar_p) = model.rssm.step(state_rssm, action)
					if stochastic:
						state_rssm = state_next
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
					f"FPS {fps:.1f}  dev={dev.type}  viz={mode}  fp16={use_half}  {'stoch' if stochastic else 'det'}  {'idle-step' if args.idle_step else 'hold'}",
					"Keys: q..y action, Shift=neg | v=viz | o=orig | j=jpg | s=stoch | d=overlay | p=dump | esc=quit",
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