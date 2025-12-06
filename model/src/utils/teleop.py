import time
from typing import Dict, Any, Tuple

from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


def setup_so101_with_camera(
    follower_port: str,
    leader_port: str,
    follower_id: str = "follower_arm",
    leader_id: str = "leader_arm",
    camera_index: int = 0,
    camera_fps: int = 30,
    camera_width: int = 1920,
    camera_height: int = 1080,
    calibrate: bool = True,
) -> Tuple[SO101Follower, SO101Leader]:
    """Create and connect SO101 follower + leader with an attached OpenCV camera.

    Returns connected (robot, teleop_device).
    """
    cam_cfg = OpenCVCameraConfig(
        index_or_path=camera_index,
        fps=camera_fps,
        width=camera_width,
        height=camera_height,
    )

    robot_cfg = SO101FollowerConfig(
        port=follower_port,
        id=follower_id,
        cameras={"front": cam_cfg},
    )
    teleop_cfg = SO101LeaderConfig(
        port=leader_port,
        id=leader_id,
    )

    robot = SO101Follower(robot_cfg)
    teleop = SO101Leader(teleop_cfg)

    teleop.connect(calibrate=calibrate)
    robot.connect(calibrate=calibrate)
    return robot, teleop


def step_teleop(robot: SO101Follower, teleop: SO101Leader) -> Dict[str, Any]:
    """Single teleop step: read observation, get action, send action, return dict."""
    observation = robot.get_observation()
    action = teleop.get_action()
    sent_action = robot.send_action(action)
    return {
        "timestamp": time.time(),
        "observation": observation,
        "action": action,
        "sent_action": sent_action,
    }


def teardown(robot: SO101Follower, teleop: SO101Leader) -> None:
    teleop.disconnect()
    robot.disconnect()
