import argparse
import os
import time
import cv2
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower


def main(
    follower_port: str,
    follower_id: str,
    leader_port: str,
    leader_id: str,
    camera_index: int = 0,
    camera_fps: int = 30,
    camera_width: int = 1920,
    camera_height: int = 1080,
    calibrate: bool = True,
):
    # Camera config attached to follower
    camera_config = {
        "front": OpenCVCameraConfig(
            index_or_path=camera_index,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        )
    }

    robot_config = SO101FollowerConfig(
        port=follower_port,
        id=follower_id,
        cameras=camera_config, # type: ignore
    )
    teleop_config = SO101LeaderConfig(
        port=leader_port,
        id=leader_id,
    )

    robot = SO101Follower(robot_config)
    teleop_device = SO101Leader(teleop_config)

    # Connect
    teleop_device.connect(calibrate=calibrate)
    robot.connect(calibrate=calibrate)

    try:
        last_ts = time.time()
        while True:
            # Read observation (camera + joints) and display camera
            observation = robot.get_observation()
            frame = observation.get("front")
            if frame is not None:
                # The OpenCV display expects BGR if frames are RGB
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Follower Camera (front)", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Teleop action and send to robot
            action = teleop_device.get_action()
            robot.send_action(action)

            # Optional loop pacing/logging
            now = time.time()
            if now - last_ts > 1.0:
                last_ts = now
            time.sleep(0.001)
    finally:
        teleop_device.disconnect()
        robot.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load .env from current folder or any parent (workspace root)
    # Prefer a .env in the script folder; otherwise fallback to nearest parent .env
    local_env = Path(__file__).with_name(".env")
    if local_env.exists():
        load_dotenv(dotenv_path=str(local_env), override=False)
    else:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(dotenv_path=env_path, override=False)

    parser = argparse.ArgumentParser(description="SO101 teleop API example with OpenCV display")
    parser.add_argument("--follower-port", default=os.environ.get("FOLLOWER_PORT"), required=os.environ.get("FOLLOWER_PORT") is None)
    parser.add_argument("--leader-port", default=os.environ.get("LEADER_PORT"), required=os.environ.get("LEADER_PORT") is None)
    parser.add_argument("--follower-id", default=os.environ.get("FOLLOWER_ID", "follower_arm"))
    parser.add_argument("--leader-id", default=os.environ.get("LEADER_ID", "leader_arm"))
    parser.add_argument("--camera-index", type=int, default=int(os.environ.get("CAMERA_INDEX", 0)))
    parser.add_argument("--camera-fps", type=int, default=int(os.environ.get("CAMERA_FPS", 30)))
    parser.add_argument("--camera-width", type=int, default=int(os.environ.get("CAMERA_WIDTH", 1920)))
    parser.add_argument("--camera-height", type=int, default=int(os.environ.get("CAMERA_HEIGHT", 1080)))
    parser.add_argument("--skip-calibrate", action="store_true")
    args = parser.parse_args()

    main(
        follower_port=args.follower_port,
        follower_id=args.follower_id,
        leader_port=args.leader_port,
        leader_id=args.leader_id,
        camera_index=args.camera_index,
        camera_fps=args.camera_fps,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        calibrate=(not args.skip_calibrate),
    )
