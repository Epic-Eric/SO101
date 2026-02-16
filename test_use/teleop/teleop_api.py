
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
    force_feedback: bool = False,
    ff_gain: float = 0.3,
    ff_threshold: float = 3.0,
    ff_max_delta: float = 10.0,
    ff_vel_move_thresh: float = 5.0,
    ff_vel_block_thresh: float = 2.0,
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
        # Simple loop with minimal feedback state (hysteresis + ramp + velocity gating)
        feedback_active = False
        ff_gain_curr = 0.0
        last_gain_time = time.time()
        prev_leader_pos: dict[str, float] | None = None
        prev_follower_pos: dict[str, float] | None = None
        prev_time = last_gain_time
        while True:
            # 1) Read leader present position and send to follower (direct position teleop)
            leader_action = teleop_device.get_action()
            robot.send_action(leader_action)

            # 2) Read latest follower observation (camera + joint positions) and display camera
            observation = robot.get_observation()
            frame = observation.get("front")
            if frame is not None:
                # The OpenCV display expects BGR if frames are RGB
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Follower Camera (front)", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 3) Optional simple force feedback: enable when error exceeds threshold, ramp gain,
            #    and gate by simple velocity heuristics to avoid false engagement when moving fast.
            if force_feedback and observation:
                # Extract joint position maps
                leader_pos = {k: float(v) for k, v in leader_action.items() if k.endswith(".pos")}
                follower_pos = {
                    k: float(v)
                    for k, v in observation.items()
                    if isinstance(v, (int, float)) and k.endswith(".pos")
                }
                feedback_goal: dict[str, float] = {}

                # Compute max absolute error across joints
                errs = []
                for k, x_m in leader_pos.items():
                    x_s = follower_pos.get(k)
                    if x_s is None:
                        continue
                    errs.append(abs(x_s - x_m))
                max_err = max(errs) if errs else 0.0

                # Simple velocities
                now_t = time.time()
                dt_vel = max(1e-3, now_t - prev_time)
                leader_speeds = []
                follower_speeds = []
                if prev_leader_pos is not None:
                    for k, x_m in leader_pos.items():
                        x_prev = prev_leader_pos.get(k, x_m)
                        leader_speeds.append(abs((x_m - x_prev) / dt_vel))
                if prev_follower_pos is not None:
                    for k, x_s in follower_pos.items():
                        x_prev = prev_follower_pos.get(k, x_s)
                        follower_speeds.append(abs((x_s - x_prev) / dt_vel))
                leader_speed_max = max(leader_speeds) if leader_speeds else 0.0
                follower_speed_max = max(follower_speeds) if follower_speeds else 0.0
                prev_leader_pos = leader_pos
                prev_follower_pos = follower_pos
                prev_time = now_t

                # Hysteresis: turn on at ff_threshold, turn off at half threshold
                dt = max(1e-3, now_t - last_gain_time)
                off_threshold = 0.5 * ff_threshold
                # Engage if large error AND (leader slow OR follower appears blocked while leader moving)
                # - leader slow condition: leader_speed_max is below move threshold -> likely contact/stop
                # - follower blocked condition: follower_speed_max below block threshold while leader moving fast
                leader_slow = leader_speed_max <= ff_vel_move_thresh
                follower_blocked = (leader_speed_max > ff_vel_move_thresh) and (follower_speed_max <= ff_vel_block_thresh)
                if not feedback_active and (max_err >= ff_threshold) and (leader_slow or follower_blocked):
                    feedback_active = True
                elif feedback_active and max_err <= off_threshold:
                    feedback_active = False

                # Gain ramp: reach ff_gain in ~0.2s, decay to 0 in ~0.1s
                RAMP_UP_SEC = 0.2
                RAMP_DOWN_SEC = 0.1
                if feedback_active:
                    ff_gain_curr = min(ff_gain, ff_gain_curr + (ff_gain / RAMP_UP_SEC) * dt)
                else:
                    ff_gain_curr = max(0.0, ff_gain_curr - (ff_gain / RAMP_DOWN_SEC) * dt)
                last_gain_time = now_t

                for k, x_m in leader_pos.items():
                    x_s = follower_pos.get(k)
                    if x_s is None:
                        continue
                    # Simple proportional feedback with ramped gain
                    pos_err = x_s - x_m
                    if ff_gain_curr <= 1e-4:
                        # Ensure torque is off when feedback is effectively disabled
                        try:
                            teleop_device.bus.disable_torque()
                        except Exception:
                            pass
                        continue
                    delta = ff_gain_curr * pos_err
                    # Clamp per-step correction to avoid aggressive jumps
                    if delta > ff_max_delta:
                        delta = ff_max_delta
                    elif delta < -ff_max_delta:
                        delta = -ff_max_delta
                    feedback_goal[k] = x_m + delta

                if feedback_goal:
                    # Enable torque when sending feedback commands
                    try:
                        teleop_device.bus.enable_torque()
                    except Exception:
                        pass
                    teleop_device.send_feedback(feedback_goal)
                else:
                    # No feedback to send; turn torque off to free the leader arm
                    try:
                        teleop_device.bus.disable_torque()
                    except Exception:
                        pass

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
    parser.add_argument("--force-feedback", action="store_true", help="Enable force feedback (write follower position back to leader)")
    parser.add_argument("--leader-port", default=os.environ.get("LEADER_PORT"), required=os.environ.get("LEADER_PORT") is None)
    parser.add_argument("--follower-id", default=os.environ.get("FOLLOWER_ID", "follower_arm"))
    parser.add_argument("--leader-id", default=os.environ.get("LEADER_ID", "leader_arm"))
    parser.add_argument("--camera-index", type=int, default=int(os.environ.get("CAMERA_INDEX", 0)))
    parser.add_argument("--camera-fps", type=int, default=int(os.environ.get("CAMERA_FPS", 30)))
    parser.add_argument("--camera-width", type=int, default=int(os.environ.get("CAMERA_WIDTH", 1920)))
    parser.add_argument("--camera-height", type=int, default=int(os.environ.get("CAMERA_HEIGHT", 1080)))
    parser.add_argument("--ff-gain", type=float, default=float(os.environ.get("FF_GAIN", 0.3)), help="Proportional feedback gain")
    parser.add_argument("--ff-threshold", type=float, default=float(os.environ.get("FF_THRESHOLD", 10.0)), help="Error threshold to trigger feedback")
    parser.add_argument("--ff-max-delta", type=float, default=float(os.environ.get("FF_MAX_DELTA", 10.0)), help="Clamp per-step correction magnitude")
    parser.add_argument("--ff-vel-move-thresh", type=float, default=float(os.environ.get("FF_VEL_MOVE_THRESH", 5.0)), help="Leader speed below which we consider it slow/stopping")
    parser.add_argument("--ff-vel-block-thresh", type=float, default=float(os.environ.get("FF_VEL_BLOCK_THRESH", 2.0)), help="Follower speed below which we consider it blocked")
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
        force_feedback=args.force_feedback,
        ff_gain=args.ff_gain,
        ff_threshold=args.ff_threshold,
        ff_max_delta=args.ff_max_delta,
        ff_vel_move_thresh=args.ff_vel_move_thresh,
        ff_vel_block_thresh=args.ff_vel_block_thresh,
        calibrate=(not args.skip_calibrate),
    )
