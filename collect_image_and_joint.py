import os
import time
from pathlib import Path
from typing import Optional

import cv2
from dotenv import load_dotenv, find_dotenv

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower


def _load_env() -> None:
	"""Load environment variables from a local .env if present, else fallback to parent .env."""
	local_env = Path(__file__).with_name(".env")
	if local_env.exists():
		load_dotenv(dotenv_path=str(local_env), override=False)
	else:
		env_path = find_dotenv(usecwd=True)
		if env_path:
			load_dotenv(dotenv_path=env_path, override=False)


def collect_synchronized_images_and_joints(
	save_dir: str = "data/captured_images_and_joints",
	camera_index: Optional[int] = None,
	camera_fps: Optional[int] = None,
	camera_width: Optional[int] = None,
	camera_height: Optional[int] = None,
	calibrate: bool = True,
) -> None:
	"""
	Continuously collect timestamped images from the follower's camera and
	timestamped joint data (for all 6 motors). Ensures one joint record per image
	with aligned timestamps. Joint data records the delta angle compared to the
	previous timestamp. Saves to `save_dir` as image files and a `joints.jsonl` manifest.

	Press Ctrl-C to stop.
	"""
	_load_env()

	follower_port = os.environ.get("FOLLOWER_PORT")
	follower_id = os.environ.get("FOLLOWER_ID", "follower_arm")

	assert follower_port is not None, "FOLLOWER_PORT must be set in environment"

	# Camera params: env overrides, then arguments, then defaults
	cam_index = int(os.environ.get("CAMERA_INDEX", camera_index if camera_index is not None else 0))
	cam_fps = int(os.environ.get("CAMERA_FPS", camera_fps if camera_fps is not None else 30))
	cam_width = int(os.environ.get("CAMERA_WIDTH", camera_width if camera_width is not None else 1920))
	cam_height = int(os.environ.get("CAMERA_HEIGHT", camera_height if camera_height is not None else 1080))

	camera_config = {
		"front": OpenCVCameraConfig(
			index_or_path=cam_index, width=cam_width, height=cam_height, fps=cam_fps
		)
	}

	robot_config = SO101FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)  # type: ignore
	robot = SO101Follower(robot_config)

	# Prepare output directory and manifest
	out_dir = Path(save_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	joints_manifest_path = out_dir / "joints.jsonl"
	# Line format: {"t": <timestamp>, "image": "frame_000001.jpg", "joints": {"shoulder_pan.pos": ..., ...}}
	joints_manifest = open(joints_manifest_path, "a", buffering=1)

	# Connect robot (and camera), then disable torque before starting capture
	robot.connect(calibrate=calibrate)
	robot.bus.disable_torque()

	frame_idx = 0
	prev_joints: dict[str, float] | None = None
	try:
		while True:
			# Single observation read: includes camera and joint positions
			obs = robot.get_observation()
			frame = obs.get("front")

			# Skip if no frame available to maintain 1:1 image <-> joint record
			if frame is None:
				time.sleep(0.001)
				continue

			# Timestamp once per observation; shared by image and joints
			ts = time.time()

			# Save image (convert RGB -> BGR for OpenCV write) and display
			bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			img_name = f"frame_{frame_idx:06d}.jpg"
			img_path = out_dir / img_name
			cv2.imwrite(str(img_path), bgr)

			# Live view window
			cv2.imshow("Follower Camera (front)", bgr)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

			# Extract joint positions for all six motors
			# Keys are of the form "<motor_name>.pos"
			joints = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
			if prev_joints is None:
				delta = {k: 0.0 for k in joints.keys()}
			else:
				# Compute per-motor angle delta vs previous sample
				delta = {k: joints.get(k, 0.0) - prev_joints.get(k, 0.0) for k in joints.keys()}

			# Write one JSONL line per image with aligned timestamp
			# Minimal, dependency-free JSON writing
			# We avoid importing json for speed; values are primitives so this is safe.
			def _escape(s: str) -> str:
				return s.replace("\\", "\\\\").replace("\"", "\\\"")

			joints_items = ", ".join([f'"{_escape(k)}": {delta[k]:.6f}' for k in sorted(delta.keys())])
			line = f'{{"t": {ts:.6f}, "image": "{_escape(img_name)}", "joints": {{{joints_items}}}}}\n'
			joints_manifest.write(line)

			frame_idx += 1
			prev_joints = joints
			# Micro-sleep to be gentle with CPU; real-time capture predominantly paced by hardware
			time.sleep(0.001)
	except KeyboardInterrupt:
		pass
	finally:
		try:
			joints_manifest.close()
		finally:
			robot.disconnect()
			cv2.destroyAllWindows()


if __name__ == "__main__":
	collect_synchronized_images_and_joints()

