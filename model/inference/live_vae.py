#!/usr/bin/env python3
"""Live VAE inference: show camera feed, reconstruction, and latent heatmap.

Placed under `model/` so it can import model package modules more reliably.

Usage: python model/live_vae.py --model <path/to/model_final.pt|checkpoint.pt> [--cam 0] [--device cpu]
"""
import argparse
import os
import time
from typing import Optional

import cv2
import numpy as np
import torch

from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from model.src.models.vae import VAE
from model.src.utils.normalization import get_default_normalization, denormalize
from model.src.utils.teleop import setup_so101_with_camera, step_teleop, teardown


def find_model_in_artifacts(output_dir: str = "output") -> Optional[str]:
    """Scan output/artifacts for the most recent run that contains a checkpoint or final model.

    Preference order per run: checkpoint_latest.pt -> checkpoint_best.pt -> model_final.pt
    Returns full path or None if nothing found.
    """
    base = os.path.join(output_dir, "artifacts")
    if not os.path.isdir(base):
        return None
    # list runs sorted by modified time (newest first)
    runs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    runs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for r in runs:
        candidates = [
            os.path.join(r, "checkpoint_latest.pt"),
            os.path.join(r, "checkpoint_best.pt"),
            os.path.join(r, "model_final.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
    return None


def load_model_weights(path: str, device: torch.device):
    data = torch.load(path, map_location=device)
    # payload may be either {'model_state': ...} or direct state_dict
    if isinstance(data, dict) and "model_state" in data:
        state = data["model_state"]
    elif isinstance(data, dict) and "model_state" not in data and any(k.startswith("decoder.") for k in data.keys()):
        state = data
    else:
        # fallback
        state = data
    # attempt to infer latent dim from decoder fc weight if present
    latent_dim = 128
    try:
        # decoder.fc.weight shape: (out_features, in_features) where in_features == latent_dim
        w = state.get("decoder.fc.weight") or state.get("decoder.fc.weight.data")
        if w is not None:
            latent_dim = w.shape[1]
    except Exception:
        pass

    # choose tanh by default because training used normalization to [-1,1]
    vae = VAE(in_channels=3, latent_dim=latent_dim, output_activation="tanh", rec_loss="mse")
    vae.load_state_dict(state)
    vae.to(device)
    vae.eval()
    return vae


def preprocess_frame(frame: np.ndarray, size=(64, 64), norm=None, device=None):
    # frame: BGR uint8
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = size
    img = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0)
    if norm is not None:
        mean = torch.tensor(norm.mean).view(1, 3, 1, 1)
        std = torch.tensor(norm.std).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def tensor_to_bgr_img(tensor: torch.Tensor, norm, upsize=None):
    # tensor: 1xCxHxW, normalized
    with torch.no_grad():
        t = denormalize(tensor.squeeze(0).cpu(), norm).clamp(0, 1)
    npimg = (t * 255).byte().permute(1, 2, 0).numpy()
    # RGB -> BGR
    bgr = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
    if upsize is not None:
        bgr = cv2.resize(bgr, upsize, interpolation=cv2.INTER_NEAREST)
    return bgr


def latent_to_heatmap(mu: torch.Tensor, size=(256, 256)):
    # mu: 1 x D
    z = mu.squeeze(0).cpu().numpy()
    length = z.shape[-1]
    side = int(np.ceil(np.sqrt(length)))
    pad = side * side - length
    if pad > 0:
        z = np.pad(z, (0, pad), mode="constant")
    img = z.reshape(side, side)
    # normalize to 0..255
    mi, ma = img.min(), img.max()
    if ma - mi > 1e-6:
        imgn = (img - mi) / (ma - mi)
    else:
        imgn = np.zeros_like(img)
    imgu = (imgn * 255).astype(np.uint8)
    imgu = cv2.resize(imgu, size, interpolation=cv2.INTER_NEAREST)
    # apply colormap
    heat = cv2.applyColorMap(imgu, cv2.COLORMAP_VIRIDIS)
    return heat


def run_live(model_path: str, cam_idx: int = 0, device_str: str = "cpu", teleop: bool = False):
    device = torch.device(device_str)
    vae = load_model_weights(model_path, device)
    norm = get_default_normalization()

    win_name = "live_vae"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    target_display = (320, 320)

    # helper to load .env like collect_images_with_teleoperation
    def _load_env_if_present() -> None:
        local = Path(__file__).with_name(".env")
        if local.exists():
            load_dotenv(dotenv_path=str(local), override=False)
        else:
            envp = find_dotenv(usecwd=True)
            if envp:
                load_dotenv(dotenv_path=envp, override=False)

    cap = None
    robot = None
    teleop_dev = None
    try:
        if cam_idx == -1:
            # special sentinel: use teleop mode (should be passed by flag); keep backward compat
            teleop_mode = True
        else:
            teleop_mode = False
        # default: use local camera
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            # camera may be unavailable; continue and rely on teleop if configured by env
            cap = None

        while True:
            # If teleop env is configured and follower/leader ports present, prefer teleop frames
            use_teleop_now = False
            follower_port = os.environ.get("FOLLOWER_PORT")
            leader_port = os.environ.get("LEADER_PORT")
            if follower_port and leader_port:
                use_teleop_now = True

            # honor explicit teleop flag
            if teleop:
                use_teleop_now = True

            if use_teleop_now:
                # ensure teleop is connected
                if robot is None or teleop_dev is None:
                    _load_env_if_present()
                    follower_port = os.environ.get("FOLLOWER_PORT")
                    leader_port = os.environ.get("LEADER_PORT")
                    follower_id = os.environ.get("FOLLOWER_ID", "follower_arm")
                    leader_id = os.environ.get("LEADER_ID", "leader_arm")
                    camera_index = int(os.environ.get("CAMERA_INDEX", 0))
                    camera_fps = int(os.environ.get("CAMERA_FPS", 30))
                    camera_width = int(os.environ.get("CAMERA_WIDTH", 1920))
                    camera_height = int(os.environ.get("CAMERA_HEIGHT", 1080))

                    # cast to str to satisfy type checkers (we already ensured they exist)
                    robot, teleop_dev = setup_so101_with_camera(
                        follower_port=str(follower_port),
                        leader_port=str(leader_port),
                        follower_id=follower_id,
                        leader_id=leader_id,
                        camera_index=camera_index,
                        camera_fps=camera_fps,
                        camera_width=camera_width,
                        camera_height=camera_height,
                        calibrate=True,
                    )

                step = step_teleop(robot, teleop_dev)
                obs = step.get("observation", {})
                frame_rgb = obs.get("front")
                action = step.get("action")
                if frame_rgb is None:
                    time.sleep(0.01)
                    continue
                # frame_rgb is expected to be RGB; convert to BGR for display
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                if cap is None:
                    time.sleep(0.01)
                    continue
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                action = None

            # preprocess
            inp = preprocess_frame(frame, size=(64, 64), norm=norm, device=device)
            with torch.no_grad():
                rec, mu, logvar = vae(inp)

            rec_img = tensor_to_bgr_img(rec, norm, upsize=target_display)
            orig = cv2.resize(frame, target_display)
            latent_hm = latent_to_heatmap(mu, size=target_display)

            # assemble horizontally
            combo = cv2.hconcat([orig, rec_img, latent_hm])


            cv2.imshow(win_name, combo)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        if cap is not None:
            cap.release()
        if robot is not None and teleop_dev is not None:
            try:
                teardown(robot, teleop_dev)
            except Exception:
                pass
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=False, help="Path to model_final.pt or checkpoint file. If omitted, script will scan output/artifacts for latest checkpoint.")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--device", default="cpu", help="Device (cpu,cuda,mps)")
    parser.add_argument("--teleop", action="store_true", help="Enable teleoperation mode", default=True)
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = find_model_in_artifacts("output")
        if model_path is None:
            raise SystemExit("No model provided and no checkpoint/model found under output/artifacts")
        print(f"No --model provided; using discovered model: {model_path}")

    run_live(model_path, cam_idx=args.cam, device_str=args.device, teleop=args.teleop)


if __name__ == "__main__":
    main()
