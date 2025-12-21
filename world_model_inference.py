import os
import random
import json
from typing import List, Tuple

import torch
import pygame
from torchvision.io import read_image
import torchvision.transforms.functional as TF

from model.src.models.world_model import WorldModel
from model.src.utils.normalization import get_default_normalization, denormalize


def _auto_device() -> torch.device:
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


def _to_surface(img: torch.Tensor, norm_params, scale: int = 6) -> pygame.Surface:
	x = denormalize(img, norm_params).clamp(0, 1)
	x = (x * 255.0).byte().cpu()
	x = x.permute(1, 2, 0).numpy()
	surf = pygame.surfarray.make_surface(x)
	w, h = surf.get_size()
	return pygame.transform.scale(surf, (w * scale, h * scale))


def main():
	data_root = os.path.join("data", "captured_images_and_joints")
	artifact_dir = os.path.join("output", "artifacts")
	ckpt_path = os.path.join(artifact_dir, "checkpoint_best.pt")

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
	model.load_state_dict(state)
	model.eval()

	imgs = []
	for name in frames:
		p = os.path.join(data_root, name)
		img = read_image(p)  # (C,H,W) uint8
		imgs.append(img)
	images = torch.stack(imgs, dim=0).to(dev)  # (T,C,H,W)
	images = _normalize_image(images, norm_params)

	images_bt = images.unsqueeze(0)  # (1,T,C,H,W)
	actions_bt = actions_seq.unsqueeze(0).to(dev)  # (1,T-1,A)

	with torch.no_grad():
		out = model(images_bt, actions_bt)
	recon0 = out.x_rec[0, 0]

	start_image = images[0].unsqueeze(0)  # (1,C,H,W)

	pygame.init()
	pygame.display.set_caption("World Model Inference")
	surface0 = _to_surface(recon0, norm_params)
	screen = pygame.display.set_mode(surface0.get_size())

	keymap = {
		pygame.K_q: 0,
		pygame.K_w: 1,
		pygame.K_e: 2,
		pygame.K_r: 3,
		pygame.K_t: 4,
		pygame.K_y: 5,
	}

	step_size = 0.2
	current_img = recon0.clone()

	running = True
	clock = pygame.time.Clock()

	with torch.no_grad():
		x_rec0, mu0, logvar0, _ = model.vae(start_image.to(dev))
		z = model.vae.reparameterize(mu0, logvar0)
		state_rssm = model.rssm.init_state(batch_size=1, device=dev)
		state_rssm = type(state_rssm)(h=state_rssm.h, z=z)

		while running:
			action = torch.zeros(1, len(action_order), device=dev)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						running = False
					idx = keymap.get(event.key, None)
					if idx is not None:
						val = -step_size if (event.mod & pygame.KMOD_SHIFT) else step_size
						action[0, idx] = val

			state_rssm, _ = model.rssm.step(state_rssm, action)
			current_img = model.vae.decode(state_rssm.z)[0]

			surf = _to_surface(current_img, norm_params)
			screen.blit(surf, (0, 0))
			pygame.display.flip()
			clock.tick(30)

	pygame.quit()


if __name__ == "__main__":
	main()

