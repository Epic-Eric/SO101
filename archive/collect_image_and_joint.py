import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from dotenv import load_dotenv, find_dotenv

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader


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
	with aligned timestamps. Saves to an episode subfolder under `save_dir` as image
	files and a `joints.jsonl` manifest.

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

	def _new_episode_dir(root: Path) -> Path:
		stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
		# include ms to avoid collisions
		stamp = f"{stamp}_{int(time.time() * 1000) % 1000:03d}"
		p = root / f"episode_{stamp}"
		p.mkdir(parents=True, exist_ok=False)
		return p

	# Prepare output directory and manifest (one episode per run)
	root_dir = Path(save_dir)
	root_dir.mkdir(parents=True, exist_ok=True)
	out_dir = _new_episode_dir(root_dir)
	joints_manifest_path = out_dir / "joints.jsonl"
	# Line format: {"t": <timestamp>, "image": "frame_000001.jpg", "joints": {"shoulder_pan.pos": ..., ...}}
	joints_manifest = open(joints_manifest_path, "a", buffering=1)
	meta_path = out_dir / "meta.json"

	# Connect robot (and camera), then disable torque before starting capture
	robot.connect(calibrate=calibrate)
	robot.bus.disable_torque()

	frame_idx = 0
	prev_joints: dict[str, float] | None = None
	prev_ts: float | None = None
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
			dt = 0.0 if prev_ts is None else float(ts - prev_ts)

			# Save image (convert RGB -> BGR for OpenCV write) and display
			bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			img_name = f"frame_{frame_idx:06d}.jpg"
			img_path = out_dir / img_name
			cv2.imwrite(str(img_path), bgr)

			# Live view window
			cv2.imshow("Follower Camera (front)", bgr)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

			# Extract joint positions for all motors
			# Keys are of the form "<motor_name>.pos"
			joints = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
			if prev_joints is None:
				joints_delta = {k: 0.0 for k in joints.keys()}
			else:
				# Compute per-motor angle delta vs previous sample
				joints_delta = {k: joints.get(k, 0.0) - prev_joints.get(k, 0.0) for k in joints.keys()}

			# Write one JSONL line per image with aligned timestamp.
			# Keep backward compatibility: `joints` contains absolute positions.
			rec = {
				"t": float(ts),
				"dt": float(dt),
				"image": img_name,
				"joints": joints,
				"joints_delta": joints_delta,
			}
			joints_manifest.write(json.dumps(rec) + "\n")

			frame_idx += 1
			prev_joints = joints
			prev_ts = ts
			# Micro-sleep to be gentle with CPU; real-time capture predominantly paced by hardware
			time.sleep(0.001)
	except KeyboardInterrupt:
		pass
	finally:
		try:
			joints_manifest.close()
		finally:
			# best-effort: write metadata
			try:
				meta = {
					"episode_dir": str(out_dir),
					"created_at": datetime.now().isoformat(),
					"camera": {"index": cam_index, "fps": cam_fps, "width": cam_width, "height": cam_height},
				}
				meta_path.write_text(json.dumps(meta, indent=2) + "\n")
			except Exception:
				pass
			robot.disconnect()
			cv2.destroyAllWindows()


if __name__ == "__main__":
	_load_env()
	# If LEADER_PORT is present, run teleop-follow + logging; else run passive logging
	leader_port = os.environ.get("LEADER_PORT")
	if leader_port:
		print("LEADER_PORT detected; running teleop-follow with logging...")
		# Teleop-follow mode
		def collect_follow_and_log(
			save_dir: str = "data/captured_images_and_joints",
			camera_index: Optional[int] = None,
			camera_fps: Optional[int] = None,
			camera_width: Optional[int] = None,
			camera_height: Optional[int] = None,
			calibrate: bool = True,
		) -> None:
			follower_port = os.environ.get("FOLLOWER_PORT")
			follower_id = os.environ.get("FOLLOWER_ID", "follower_arm")
			leader_id = os.environ.get("LEADER_ID", "leader_arm")

			assert follower_port is not None, "FOLLOWER_PORT must be set in environment"
			assert leader_port is not None, "LEADER_PORT must be set in environment"

			cam_index = int(os.environ.get("CAMERA_INDEX", camera_index if camera_index is not None else 0))
			cam_fps = int(os.environ.get("CAMERA_FPS", camera_fps if camera_fps is not None else 30))
			cam_width = int(os.environ.get("CAMERA_WIDTH", camera_width if camera_width is not None else 1920))
			cam_height = int(os.environ.get("CAMERA_HEIGHT", camera_height if camera_height is not None else 1080))

			camera_config = {
				"front": OpenCVCameraConfig(index_or_path=cam_index, width=cam_width, height=cam_height, fps=cam_fps)
			}

			robot_config = SO101FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)  # type: ignore
			teleop_config = SO101LeaderConfig(port=leader_port, id=leader_id)

			robot = SO101Follower(robot_config)
			teleop_device = SO101Leader(teleop_config)

			def _new_episode_dir(root: Path) -> Path:
				stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
				stamp = f"{stamp}_{int(time.time() * 1000) % 1000:03d}"
				p = root / f"episode_{stamp}"
				p.mkdir(parents=True, exist_ok=False)
				return p

			def _action_to_vec6(action: object, joint_keys6: list[str]) -> list[float]:
				# Robust mapping: match exact joint keys, or key without ".pos".
				if action is None:
					return [0.0] * 6
				if isinstance(action, (list, tuple)):
					vals = [float(x) for x in action]
					if len(vals) == 6:
						return vals
					# pad/trim if unexpected
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
					# clip to a sane range
					if x > 1.0:
						x = 1.0
					elif x < -1.0:
						x = -1.0
					out.append(x)
				return (out + [0.0] * 6)[:6]

			# Output directory and manifest (one episode per run)
			root_dir = Path(save_dir)
			root_dir.mkdir(parents=True, exist_ok=True)
			out_dir = _new_episode_dir(root_dir)
			joints_manifest_path = out_dir / "joints.jsonl"
			joints_manifest = open(joints_manifest_path, "a", buffering=1)
			meta_path = out_dir / "meta.json"

			teleop_device.connect(calibrate=calibrate)
			robot.connect(calibrate=calibrate)
			# Ensure torque is enabled for following
			robot.bus.enable_torque()

			frame_idx = 0
			prev_joints: dict[str, float] | None = None
			prev_ts: float | None = None
			joint_keys6: list[str] | None = None
			action_scale = float(os.environ.get("ACTION_SCALE", "1.0"))
			try:
				while True:
					# Get leader action and send to follower
					action = teleop_device.get_action()
					robot.send_action(action)

					# Read observation (camera + joints) from follower
					obs = robot.get_observation()
					frame = obs.get("front")
					if frame is None:
						time.sleep(0.001)
						continue

					# Timestamp
					ts = time.time()
					dt = 0.0 if prev_ts is None else float(ts - prev_ts)

					# Save image (convert RGB -> BGR for OpenCV write) and display
					bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
					img_name = f"frame_{frame_idx:06d}.jpg"
					img_path = out_dir / img_name
					cv2.imwrite(str(img_path), bgr)

					cv2.imshow("Follower Camera (front)", bgr)
					if cv2.waitKey(1) & 0xFF == ord("q"):
						break

					# Extract joint positions and compute deltas
					joints = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
					if joint_keys6 is None:
						# Fixed 6D order for action vectorization
						joint_keys6 = sorted(joints.keys())[:6]
						try:
							meta = {
								"episode_dir": str(out_dir),
								"created_at": datetime.now().isoformat(),
								"camera": {"index": cam_index, "fps": cam_fps, "width": cam_width, "height": cam_height},
								"joint_keys6": joint_keys6,
								"action_scale": action_scale,
							}
							meta_path.write_text(json.dumps(meta, indent=2) + "\n")
						except Exception:
							pass

					if prev_joints is None:
						joints_delta = {k: 0.0 for k in joints.keys()}
					else:
						joints_delta = {k: joints.get(k, 0.0) - prev_joints.get(k, 0.0) for k in joints.keys()}

					vec6_raw = _action_to_vec6(action, joint_keys6 or sorted(joints.keys())[:6])
					action6 = _normalize_vec(vec6_raw, action_scale)

					rec = {
						"t": float(ts),
						"dt": float(dt),
						"image": img_name,
						"joints": joints,
						"joints_delta": joints_delta,
						"action6": action6,
					}
					joints_manifest.write(json.dumps(rec) + "\n")

					frame_idx += 1
					prev_joints = joints
					prev_ts = ts
					time.sleep(0.001)
			finally:
				try:
					joints_manifest.close()
				finally:
					teleop_device.disconnect()
					robot.disconnect()
					cv2.destroyAllWindows()

		# Run
		collect_follow_and_log()
	else:
		collect_synchronized_images_and_joints()

