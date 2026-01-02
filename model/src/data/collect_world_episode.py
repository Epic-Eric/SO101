import os
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Callable, Optional, Any

from dotenv import load_dotenv, find_dotenv

import cv2

from model.src.utils.teleop import setup_so101_with_camera, step_teleop, teardown


def _load_env() -> None:
    local = Path(__file__).with_name(".env")
    if local.exists():
        load_dotenv(dotenv_path=str(local), override=False)
    else:
        envp = find_dotenv(usecwd=True)
        if envp:
            load_dotenv(dotenv_path=envp, override=False)


def _new_episode_dir(root: Path, episode_name: Optional[str] = None) -> Path:
    if episode_name:
        safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in episode_name)
        p = root / safe
    else:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        stamp = f"{stamp}_{int(time.time() * 1000) % 1000:03d}"
        p = root / f"episode_{stamp}"

    p.mkdir(parents=True, exist_ok=False)
    return p


def _action_to_vec6(action: object, joint_keys6: list[str]) -> list[float]:
    if action is None:
        return [0.0] * 6

    if isinstance(action, (list, tuple)):
        vals = [float(x) for x in action]
        return (vals + [0.0] * 6)[:6]

    if isinstance(action, dict):
        vec: list[float] = []
        for k in joint_keys6:
            k2 = k[:-4] if k.endswith(".pos") else k
            v = action.get(k)
            if v is None:
                v = action.get(k2)
            try:
                vec.append(float(v) if v is not None else 0.0)
            except Exception:
                vec.append(0.0)
        return (vec + [0.0] * 6)[:6]

    return [0.0] * 6


def _normalize_vec(vec6: list[float], scale: float) -> list[float]:
    s = float(scale) if scale and scale > 0 else 1.0
    out = []
    for v in vec6[:6]:
        x = float(v) / s
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0
        out.append(x)
    return (out + [0.0] * 6)[:6]


@dataclass
class EpisodeResult:
    episode_dir: str
    frames: int
    duration_sec: float


def collect_world_model_episode_teleop(
    save_root: str,
    episode_name: Optional[str] = None,
    action_scale: float = 1.0,
    calibrate: bool = True,
    camera_index: Optional[int] = None,
    camera_fps: Optional[int] = None,
    camera_width: Optional[int] = None,
    camera_height: Optional[int] = None,
    max_frames: Optional[int] = None,
    duration_sec: Optional[float] = None,
    stop_event: Optional[Event] = None,
    on_frame: Optional[Callable[[Any, int, float], None]] = None,
) -> EpisodeResult:
    """Collect one teleop episode into a fresh folder.

    Writes:
      - frame_XXXXXX.jpg
      - joints.jsonl lines with {t, dt, image, joints, joints_delta, action6}
      - meta.json with joint_keys6, action_scale, camera metadata

    `on_frame(frame_rgb, frame_idx, elapsed_sec)` is called best-effort.
    """
    _load_env()

    follower_port = os.environ.get("FOLLOWER_PORT")
    leader_port = os.environ.get("LEADER_PORT")
    follower_id = os.environ.get("FOLLOWER_ID", "follower_arm")
    leader_id = os.environ.get("LEADER_ID", "leader_arm")

    cam_index = int(os.environ.get("CAMERA_INDEX", 0)) if camera_index is None else int(camera_index)
    cam_fps = int(os.environ.get("CAMERA_FPS", 30)) if camera_fps is None else int(camera_fps)
    cam_width = int(os.environ.get("CAMERA_WIDTH", 1920)) if camera_width is None else int(camera_width)
    cam_height = int(os.environ.get("CAMERA_HEIGHT", 1080)) if camera_height is None else int(camera_height)

    assert follower_port is not None, "FOLLOWER_PORT must be set in environment"
    assert leader_port is not None, "LEADER_PORT must be set in environment"

    root = Path(save_root)
    root.mkdir(parents=True, exist_ok=True)
    out_dir = _new_episode_dir(root, episode_name=episode_name)

    joints_fp = (out_dir / "joints.jsonl").open("a", buffering=1)
    meta_path = out_dir / "meta.json"

    robot, teleop = setup_so101_with_camera(
        follower_port=follower_port,
        leader_port=leader_port,
        follower_id=follower_id,
        leader_id=leader_id,
        camera_index=cam_index,
        camera_fps=cam_fps,
        camera_width=cam_width,
        camera_height=cam_height,
        calibrate=calibrate,
    )

    frame_idx = 0
    prev_joints: dict[str, float] | None = None
    prev_ts: float | None = None
    joint_keys6: list[str] | None = None
    start = time.time()

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            step = step_teleop(robot, teleop)
            ts = float(step["timestamp"])
            obs = step["observation"]
            action = step["action"]
            frame = obs.get("front")

            if frame is None:
                time.sleep(0.001)
                continue

            dt = 0.0 if prev_ts is None else float(ts - prev_ts)

            # Save frame
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_name = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_dir / img_name), bgr)

            joints = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
            if joint_keys6 is None:
                joint_keys6 = sorted(joints.keys())[:6]
                meta = {
                    "episode_dir": str(out_dir),
                    "created_at": datetime.now().isoformat(),
                    "camera": {"index": cam_index, "fps": cam_fps, "width": cam_width, "height": cam_height},
                    "joint_keys6": joint_keys6,
                    "action_scale": float(action_scale),
                }
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")

            if prev_joints is None:
                joints_delta = {k: 0.0 for k in joints.keys()}
            else:
                joints_delta = {k: joints.get(k, 0.0) - prev_joints.get(k, 0.0) for k in joints.keys()}

            vec6_raw = _action_to_vec6(action, joint_keys6 or sorted(joints.keys())[:6])
            action6 = _normalize_vec(vec6_raw, float(action_scale))

            rec = {
                "t": ts,
                "dt": dt,
                "image": img_name,
                "joints": joints,
                "joints_delta": joints_delta,
                "action6": action6,
            }
            joints_fp.write(json.dumps(rec) + "\n")

            elapsed = time.time() - start
            if on_frame is not None:
                try:
                    on_frame(frame, frame_idx, float(elapsed))
                except Exception:
                    pass

            frame_idx += 1
            prev_joints = joints
            prev_ts = ts

            if max_frames is not None and frame_idx >= int(max_frames):
                break
            if duration_sec is not None and elapsed >= float(duration_sec):
                break

            time.sleep(0.001)
    finally:
        try:
            joints_fp.close()
        finally:
            teardown(robot, teleop)

    return EpisodeResult(episode_dir=str(out_dir), frames=frame_idx, duration_sec=float(time.time() - start))
