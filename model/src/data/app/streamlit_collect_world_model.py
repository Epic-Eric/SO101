import os
import json
import time
import queue
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Thread, Lock
from typing import Optional, Any

import cv2
import numpy as np
import streamlit as st

from dotenv import load_dotenv, find_dotenv

from model.src.utils.teleop import setup_so101_with_camera, step_teleop, teardown


def _find_workspace_root() -> Path:
    """Best-effort workspace root finder.

    Streamlit's cwd can vary depending on how it's launched. We want paths like
    `data/world_model_episodes` to resolve relative to the repo root (so101).
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "run_app.py").is_file() and (parent / "requirements.txt").is_file():
            return parent
    return Path.cwd().resolve()


_WORKSPACE_ROOT = _find_workspace_root()


def _image_fit(placeholder, frame_rgb: np.ndarray) -> None:
    # Streamlit (2026+) prefers `width='stretch'` over `use_container_width`.
    fn = getattr(placeholder, "image", None)
    if fn is None:
        return

    # Ensure sane dtype/range to avoid intermittent black frames.
    try:
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
    except Exception:
        pass

    try:
        # New API
        fn(frame_rgb, channels="RGB", width="stretch")
        return
    except TypeError:
        pass

    try:
        # Older API
        fn(frame_rgb, channels="RGB", use_container_width=True)
        return
    except TypeError:
        pass

    fn(frame_rgb, channels="RGB")


# Load environment variables from workspace .env (if present)
_env_path = find_dotenv(usecwd=True)
if _env_path:
    load_dotenv(dotenv_path=_env_path, override=False)


@dataclass
class LiveState:
    running: bool = False  # recording
    frames: int = 0
    elapsed: float = 0.0
    episode_dir: Optional[str] = None
    last_error: Optional[str] = None


@dataclass
class TeleopState:
    connected: bool = False
    fps_est: float = 0.0
    last_ok: bool = False
    last_ts: float = 0.0


def _ensure_state() -> None:
    if "live" not in st.session_state:
        st.session_state.live = LiveState()
    if "teleop_state" not in st.session_state:
        st.session_state.teleop_state = TeleopState()
    if "robot" not in st.session_state:
        st.session_state.robot = None
    if "teleop" not in st.session_state:
        st.session_state.teleop = None
    if "teleop_thread" not in st.session_state:
        st.session_state.teleop_thread = None
    if "teleop_queue" not in st.session_state:
        st.session_state.teleop_queue = queue.Queue(maxsize=1)
    if "record_queue" not in st.session_state:
        # Buffered so recorder doesn't block teleop thread
        st.session_state.record_queue = queue.Queue(maxsize=200)
    if "teleop_error_queue" not in st.session_state:
        st.session_state.teleop_error_queue = queue.Queue()
    if "teleop_stop_event" not in st.session_state:
        st.session_state.teleop_stop_event = Event()

    if "record_thread" not in st.session_state:
        st.session_state.record_thread = None
    if "record_stop_event" not in st.session_state:
        st.session_state.record_stop_event = Event()

    if "lock" not in st.session_state:
        st.session_state.lock = Lock()
    if "camera_candidates" not in st.session_state:
        st.session_state.camera_candidates = []
    if "last_preview_frame" not in st.session_state:
        st.session_state.last_preview_frame = None


def _scan_cameras(max_index: int = 10) -> list[int]:
    # Best-effort OpenCV index scan (useful for picking CAMERA_INDEX)
    found: list[int] = []
    for idx in range(int(max_index) + 1):
        cap = cv2.VideoCapture(idx)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append(idx)
        finally:
            cap.release()
    return found


def _teleop_worker(
    robot: Any,
    teleop: Any,
    preview_q: "queue.Queue",
    record_q: "queue.Queue",
    stop_event: Event,
    err_q: "queue.Queue",
    state: TeleopState,
) -> None:
    consecutive_errors = 0
    max_errors = 20
    t0 = time.time()
    frames = 0

    while not stop_event.is_set():
        try:
            data = step_teleop(robot, teleop)
            consecutive_errors = 0
            frames += 1
            now = time.time()
            dt = now - t0
            if dt > 0.5:
                state.fps_est = float(frames / max(dt, 1e-6))
                t0 = now
                frames = 0

            state.last_ok = True
            state.last_ts = float(data.get("timestamp", 0.0))

            if preview_q.full():
                try:
                    preview_q.get_nowait()
                except queue.Empty:
                    pass

            try:
                preview_q.put_nowait(data)
            except Exception:
                pass

            # Recorder queue is buffered; drop oldest on overflow.
            if record_q.full():
                try:
                    record_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                record_q.put_nowait(data)
            except Exception:
                pass
            time.sleep(0.005)
        except Exception as e:
            consecutive_errors += 1
            state.last_ok = False
            if consecutive_errors >= max_errors:
                msg = f"Teleop thread failed after {max_errors} attempts. Last error: {e}"
                err_q.put(msg)
                break
            time.sleep(0.05)

    # Emergency teardown if we crashed
    if consecutive_errors >= max_errors:
        try:
            teardown(robot, teleop)
        except Exception:
            pass


def _connect_robot(
    follower_port: str,
    leader_port: str,
    follower_id: str,
    leader_id: str,
    camera_index: int,
    camera_fps: int,
    camera_width: int,
    camera_height: int,
) -> None:
    state: TeleopState = st.session_state.teleop_state
    if state.connected:
        return

    st.session_state.teleop_stop_event.clear()
    robot, teleop = setup_so101_with_camera(
        follower_port=follower_port,
        leader_port=leader_port,
        follower_id=follower_id,
        leader_id=leader_id,
        camera_index=camera_index,
        camera_fps=camera_fps,
        camera_width=camera_width,
        camera_height=camera_height,
        calibrate=True,
    )

    st.session_state.robot = robot
    st.session_state.teleop = teleop
    state.connected = True

    t = Thread(
        target=_teleop_worker,
        args=(
            robot,
            teleop,
            st.session_state.teleop_queue,
            st.session_state.record_queue,
            st.session_state.teleop_stop_event,
            st.session_state.teleop_error_queue,
            state,
        ),
        daemon=True,
    )
    st.session_state.teleop_thread = t
    t.start()


def _disconnect_robot() -> None:
    state: TeleopState = st.session_state.teleop_state
    st.session_state.teleop_stop_event.set()
    st.session_state.record_stop_event.set()

    t = st.session_state.teleop_thread
    if t is not None and t.is_alive():
        t.join(timeout=2.0)

    rt = st.session_state.record_thread
    if rt is not None and rt.is_alive():
        rt.join(timeout=2.0)

    robot = st.session_state.robot
    teleop = st.session_state.teleop
    if robot is not None and teleop is not None:
        try:
            teardown(robot, teleop)
        except Exception:
            pass

    st.session_state.robot = None
    st.session_state.teleop = None
    st.session_state.teleop_thread = None
    st.session_state.record_thread = None
    state.connected = False


def _new_episode_dir(root: Path, episode_name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if episode_name.strip():
        safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in episode_name.strip())
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
    out: list[float] = []
    for v in vec6[:6]:
        x = float(v) / s
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0
        out.append(x)
    return (out + [0.0] * 6)[:6]


def _start_recording(
    save_root: str,
    episode_name: str,
    action_scale: float,
    max_frames: Optional[int],
    duration_sec: Optional[float],
    camera_index: int,
    camera_fps: int,
    camera_width: int,
    camera_height: int,
) -> None:
    state: TeleopState = st.session_state.teleop_state
    live: LiveState = st.session_state.live
    if not state.connected or live.running:
        return

    st.session_state.record_stop_event.clear()

    # Resolve save_root relative to workspace root if it's not absolute.
    try:
        save_root_path = Path(save_root)
        if not save_root_path.is_absolute():
            save_root_path = (_WORKSPACE_ROOT / save_root_path)
        save_root_path = save_root_path.resolve()

        out_dir = _new_episode_dir(save_root_path, episode_name)
        joints_fp = (out_dir / "joints.jsonl").open("a", buffering=1)
        meta_path = out_dir / "meta.json"
    except Exception as e:
        live.running = False
        live.episode_dir = None
        live.last_error = f"Failed to start recording: {e}"
        return

    live.running = True
    live.frames = 0
    live.elapsed = 0.0
    live.episode_dir = str(out_dir)
    live.last_error = None

    # IMPORTANT: Avoid reading st.session_state inside background threads.
    record_q = st.session_state.record_queue
    record_stop_event = st.session_state.record_stop_event

    def worker() -> None:
        start = time.time()
        frame_idx = 0
        prev_ts: float | None = None
        prev_joints: dict[str, float] | None = None
        joint_keys6: list[str] | None = None

        try:
            while not record_stop_event.is_set():
                try:
                    data = record_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                ts = float(data.get("timestamp", time.time()))
                obs = data.get("observation", {})
                action = data.get("action", None)
                frame = obs.get("front")
                if frame is None:
                    continue

                dt = 0.0 if prev_ts is None else float(ts - prev_ts)

                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                img_name = f"frame_{frame_idx:06d}.jpg"
                ok = cv2.imwrite(str(out_dir / img_name), bgr)
                if not ok:
                    raise RuntimeError(f"cv2.imwrite failed for {out_dir / img_name}")

                joints = {k: float(v) for k, v in obs.items() if isinstance(k, str) and k.endswith(".pos")}
                if joint_keys6 is None:
                    joint_keys6 = sorted(joints.keys())[:6]
                    meta = {
                        "episode_dir": str(out_dir),
                        "created_at": datetime.now().isoformat(),
                        "ports": {"follower": os.getenv("FOLLOWER_PORT"), "leader": os.getenv("LEADER_PORT")},
                        "camera": {"index": int(camera_index), "fps": int(camera_fps), "width": int(camera_width), "height": int(camera_height)},
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

                frame_idx += 1
                prev_ts = ts
                prev_joints = joints

                live.frames = frame_idx
                live.elapsed = float(time.time() - start)

                if max_frames is not None and frame_idx >= int(max_frames):
                    break
                if duration_sec is not None and live.elapsed >= float(duration_sec):
                    break
        except Exception as e:
            live.last_error = str(e)
        finally:
            try:
                joints_fp.close()
            except Exception:
                pass
            live.running = False

    t = Thread(target=worker, daemon=True)
    st.session_state.record_thread = t
    t.start()
def main() -> None:
    st.set_page_config(page_title="World Model Data Collector", layout="wide")
    _ensure_state()

    live: LiveState = st.session_state.live
    teleop_state: TeleopState = st.session_state.teleop_state

    st.title("World Model Data Collector")

    with st.sidebar:
        st.header("Teleoperation Setup")
        default_follower_port = os.getenv("FOLLOWER_PORT", "")
        default_leader_port = os.getenv("LEADER_PORT", "")
        default_follower_id = os.getenv("FOLLOWER_ID", "follower_arm")
        default_leader_id = os.getenv("LEADER_ID", "leader_arm")

        follower_port = st.text_input("Follower Port", value=default_follower_port)
        leader_port = st.text_input("Leader Port", value=default_leader_port)
        follower_id = st.text_input("Follower ID", value=default_follower_id)
        leader_id = st.text_input("Leader ID", value=default_leader_id)

        st.subheader("Camera")
        max_scan = st.number_input("Scan max index", min_value=1, max_value=20, value=10, step=1)
        if st.button("Scan cameras", disabled=teleop_state.connected or live.running):
            st.session_state.camera_candidates = _scan_cameras(max_index=int(max_scan))

        candidates: list[int] = st.session_state.camera_candidates
        default_idx = int(os.getenv("CAMERA_INDEX", "0"))
        if candidates:
            default_i = candidates.index(default_idx) if default_idx in candidates else 0
            cam_idx = st.selectbox("Camera index", options=candidates, index=default_i)
        else:
            cam_idx = st.number_input("Camera index", min_value=0, max_value=20, value=int(default_idx), step=1)

        cam_fps = st.number_input("Camera FPS", min_value=1, max_value=120, value=int(os.getenv("CAMERA_FPS", "30")), step=1)
        cam_width = st.number_input("Camera width", min_value=160, max_value=3840, value=int(os.getenv("CAMERA_WIDTH", "1920")), step=10)
        cam_height = st.number_input("Camera height", min_value=120, max_value=2160, value=int(os.getenv("CAMERA_HEIGHT", "1080")), step=10)

        st.subheader("Connection")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Connect", disabled=teleop_state.connected or live.running):
                _connect_robot(
                    follower_port=follower_port,
                    leader_port=leader_port,
                    follower_id=follower_id,
                    leader_id=leader_id,
                    camera_index=int(cam_idx),
                    camera_fps=int(cam_fps),
                    camera_width=int(cam_width),
                    camera_height=int(cam_height),
                )
        with c2:
            if st.button("Disconnect", disabled=not teleop_state.connected):
                _disconnect_robot()

        st.divider()
        st.header("Episode")
        default_save_root = os.getenv("WORLD_MODEL_SAVE_ROOT", "data/world_model_episodes")
        save_root = st.text_input("Save root", value=default_save_root)
        try:
            resolved = Path(save_root)
            if not resolved.is_absolute():
                resolved = (_WORKSPACE_ROOT / resolved)
            st.caption(f"Resolved save root: {resolved.resolve()}")
        except Exception:
            pass
        episode_name = st.text_input("Episode name (optional)", value="")
        action_scale = st.number_input("Action scale (normalize to [-1,1])", value=1.0, min_value=1e-6, step=0.1)
        max_frames = st.number_input("Max frames (0 = unlimited)", value=0, min_value=0, step=100)
        duration_sec = st.number_input("Max duration sec (0 = unlimited)", value=0.0, min_value=0.0, step=5.0)

        r1, r2 = st.columns(2)
        with r1:
            if st.button("Start recording", disabled=(not teleop_state.connected) or live.running):
                _start_recording(
                    save_root=save_root,
                    episode_name=episode_name,
                    action_scale=float(action_scale),
                    max_frames=(int(max_frames) if int(max_frames) > 0 else None),
                    duration_sec=(float(duration_sec) if float(duration_sec) > 0 else None),
                    camera_index=int(cam_idx),
                    camera_fps=int(cam_fps),
                    camera_width=int(cam_width),
                    camera_height=int(cam_height),
                )
        with r2:
            if st.button("Stop recording", disabled=not live.running):
                st.session_state.record_stop_event.set()

    tab_preview, tab_status = st.tabs(["Preview", "Status"])

    with tab_preview:
        st.subheader("Video")
        placeholder = st.empty()

        # Drain preview queue to get the newest frame without starving it.
        last = st.session_state.get("_last_preview_data")
        while True:
            try:
                last = st.session_state.teleop_queue.get_nowait()
            except queue.Empty:
                break
        if last is not None:
            st.session_state._last_preview_data = last
        frame = None
        if last is not None:
            obs = last.get("observation", {})
            frame = obs.get("front")
        if frame is not None:
            # Filter out obvious "bad" frames to reduce flicker (all-black / near-black).
            try:
                arr = frame
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    # If it's nearly black and we already have a frame, don't replace it.
                    if st.session_state.last_preview_frame is not None:
                        m = float(arr.mean())
                        s = float(arr.std())
                        if m < 1.0 and s < 1.0:
                            arr = None
                if arr is not None:
                    st.session_state.last_preview_frame = arr
            except Exception:
                st.session_state.last_preview_frame = frame

        show = st.session_state.last_preview_frame
        if show is not None:
            _image_fit(placeholder, show)
        elif not teleop_state.connected:
            st.info("Connect to robot to see live preview.")

    with tab_status:
        st.subheader("Connection")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Connected", "Yes" if teleop_state.connected else "No")
        c2.metric("Teleop OK", "Yes" if teleop_state.last_ok else "No")
        c3.metric("Teleop FPS (est)", f"{teleop_state.fps_est:.1f}")
        c4.metric("Last ts", f"{teleop_state.last_ts:.3f}")

        # Show any background thread error
        try:
            err = st.session_state.teleop_error_queue.get_nowait()
            st.error(err)
        except queue.Empty:
            pass

        st.subheader("Recording")
        r1, r2, r3 = st.columns(3)
        r1.metric("Recording", "Yes" if live.running else "No")
        r2.metric("Frames", str(live.frames))
        r3.metric("Elapsed (s)", f"{live.elapsed:.2f}")
        if live.episode_dir:
            st.write(f"Episode dir: {live.episode_dir}")
        if live.last_error:
            st.error(live.last_error)

    # Periodic rerun to keep preview/status updating while recording.
    # This avoids long blocking loops so the Stop button stays responsive.
    if live.running or teleop_state.connected:
        time.sleep(0.1)
        _rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
        if callable(_rerun):
            _rerun()


if __name__ == "__main__":
    main()
