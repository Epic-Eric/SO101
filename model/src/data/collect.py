import os
import cv2
import time
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv, find_dotenv

from ..interfaces.teleop import TeleopSample, TeleopSessionResult
from ..utils.teleop import setup_so101_with_camera, step_teleop, teardown
from ..utils import write_frame, append_actions_jsonl


def _load_env() -> None:
    # Prefer local .env then any parent .env
    local = Path(__file__).with_name(".env")
    if local.exists():
        load_dotenv(dotenv_path=str(local), override=False)
    else:
        envp = find_dotenv(usecwd=True)
        if envp:
            load_dotenv(dotenv_path=envp, override=False)


def collect_images_with_teleoperation(
    duration_sec: Optional[float] = None,
    max_frames: Optional[int] = None,
    save_dir: Optional[str] = None,
    show_window: bool = True,
) -> TeleopSessionResult:
    """
    Collect images and actions while teleoperating SO101.

    Returns TeleopSessionResult with samples and metadata.
    """
    _load_env()
    follower_port = os.environ.get("FOLLOWER_PORT")
    leader_port = os.environ.get("LEADER_PORT")
    follower_id = os.environ.get("FOLLOWER_ID", "follower_arm")
    leader_id = os.environ.get("LEADER_ID", "leader_arm")
    camera_index = int(os.environ.get("CAMERA_INDEX", 0))
    camera_fps = int(os.environ.get("CAMERA_FPS", 30))
    camera_width = int(os.environ.get("CAMERA_WIDTH", 1920))
    camera_height = int(os.environ.get("CAMERA_HEIGHT", 1080))

    assert follower_port is not None, "FOLLOWER_PORT must be set in environment"
    assert leader_port is not None, "LEADER_PORT must be set in environment"

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

    # Prepare continuous saving if requested
    output_dir = None
    actions_fp = None
    if save_dir:
        output_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        actions_fp = open(Path(save_dir) / "actions.jsonl", "a", buffering=1)

    samples: List[TeleopSample] = []
    start = time.time()
    frames_collected = 0

    try:
        while True:
            step = step_teleop(robot, teleop)
            ts = step["timestamp"]
            obs = step["observation"]
            action = step["action"]
            frame = obs.get("front")

            if show_window and frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("SO101 Front Camera", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Continuous save: image and action
            if save_dir and frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                write_frame(Path(save_dir) / f"frame_{frames_collected:06d}.jpg", bgr)
            if save_dir:
                append_actions_jsonl(Path(save_dir) / "actions.jsonl", ts, action)

            samples.append(TeleopSample(timestamp=ts, action=action, observation=obs))
            frames_collected += 1

            if duration_sec is not None and (time.time() - start) >= duration_sec:
                break
            if max_frames is not None and frames_collected >= max_frames:
                break
            time.sleep(0.001)
    finally:
        teardown(robot, teleop)

    meta = {
        "frames_collected": frames_collected,
        "duration_sec": time.time() - start,
        "camera": {
            "index": camera_index,
            "fps": camera_fps,
            "width": camera_width,
            "height": camera_height,
        },
    }
    return TeleopSessionResult(samples=samples, metadata=meta, output_dir=output_dir)
