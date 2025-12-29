import os
import time
import math
import random
import re
from typing import Dict, List, Tuple

try:
    import pybullet as p
    import pybullet_data
except Exception as e:
    raise RuntimeError("PyBullet is required. Install with: conda install -y -c conda-forge pybullet") from e


def get_sim_paths() -> Tuple[str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(here, "URDF_so101.urdf")
    return here, urdf_path

def load_world() -> Tuple[int, Dict[str, int], Dict[int, Tuple[float, float]]]:
    """Connect GUI, load plane and URDF, return body id, joint name->index, limits."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.1])
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.loadURDF("plane.urdf")

    sim_dir, urdf = get_sim_paths()
    # Ensure relative mesh references work by switching CWD to script dir
    os.chdir(sim_dir)

    # Use compatible flags if available in this PyBullet build
    flags = 0
    if hasattr(p, "URDF_MERGE_FIXED_LINKS"):
        flags |= p.URDF_MERGE_FIXED_LINKS
    # Load original URDF and surface clear errors if assets are missing
    try:
        # Level the base on the world plane
        robot_id = p.loadURDF(urdf, basePosition=[0, 0, -0.01], useFixedBase=True, flags=flags)
    except Exception as e:
        raise RuntimeError(
            "Failed to load URDF. Ensure all referenced STL/mesh files exist "
            "relative to the URDF and paths are correct."
        ) from e

    # Map joint names to indices and collect limits
    name_to_index: Dict[str, int] = {}
    limits: Dict[int, Tuple[float, float]] = {}
    for j in range(p.getNumJoints(robot_id)):
        ji = p.getJointInfo(robot_id, j)
        name = ji[1].decode("utf-8")
        joint_type = ji[2]
        if joint_type == p.JOINT_REVOLUTE:
            name_to_index[name] = j
            lower = ji[8]
            upper = ji[9]
            if lower >= upper:  # fallback if limits are not set
                lower, upper = -math.pi, math.pi
            limits[j] = (lower, upper)
            # Enable position control initially at 0
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=2.0)
        else:
            # Disable motors on non-revolute joints
            p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

    expected = [str(i) for i in range(1, 7)]
    missing_expected = [n for n in expected if n not in name_to_index]
    if missing_expected:
        print("Warning: expected joint names not found:")
        for n in missing_expected:
            print(f"  - {n}")

    return robot_id, name_to_index, limits


def make_target_ui(bounds=(0.3, 0.3, 0.3)):
    """Create sliders for target point, a Randomize button, and a visible sphere marker with axis lines."""
    bx, by, bz = bounds
    x_id = p.addUserDebugParameter("Target X", -bx, bx, 0.0)
    y_id = p.addUserDebugParameter("Target Y", -by, by, 0.0)
    z_id = p.addUserDebugParameter("Target Z", 0.0, bz, 0.15)
    rand_id = p.addUserDebugParameter("Randomize Target", 0, 1, 0)

    # Increase radius and add collision shape to make target clearly visible
    radius = 0.06
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0.1, 0.1, 1])
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    target_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[0, 0, 0.15])

    # Add axis debug lines centered at target for visibility
    L = 0.06
    center = [0, 0, 0.1]
    x_line = p.addUserDebugLine([center[0] - L, center[1], center[2]], [center[0] + L, center[1], center[2]], [1, 0, 0], lineWidth=2)
    y_line = p.addUserDebugLine([center[0], center[1] - L, center[2]], [center[0], center[1] + L, center[2]], [0, 1, 0], lineWidth=2)
    z_line = p.addUserDebugLine([center[0], center[1], center[2] - L], [center[0], center[1], center[2] + L], [0, 0, 1], lineWidth=2)

    return (x_id, y_id, z_id), rand_id, target_id, (x_line, y_line, z_line), bounds


def read_target(slider_ids: Tuple[int, int, int]) -> List[float]:
    x_id, y_id, z_id = slider_ids
    tx = p.readUserDebugParameter(x_id)
    ty = p.readUserDebugParameter(y_id)
    tz = p.readUserDebugParameter(z_id)
    return [tx, ty, tz]


def random_target(bounds: Tuple[float, float, float]) -> List[float]:
    bx, by, bz = bounds
    return [random.uniform(-bx, bx), random.uniform(-by, by), random.uniform(0.0, bz)]


def help_text():
    print("\nControls:")
    print("  1-6: increase joint angle for joints named '1'..'6'")
    print("  Shift+1..Shift+6: decrease joint angle (reverse)")
    print("  Mouse + Shift: rotate camera (drag)")
    print("  Mouse + Ctrl: pan camera (drag)")
    print("  R: randomize target point within bounds")
    print("\nGUI: Target X/Y/Z sliders and 'Randomize Target' button.")


def main():
    robot_id, name_to_index, limits = load_world()
    # Expected joints named "1".."6" from URDF
    joint_names = [str(i) for i in range(1, 7)]
    joint_indices = [name_to_index.get(n, None) for n in joint_names]

    slider_ids, rand_id, target_id, axis_ids, bounds = make_target_ui()

    # Track which joints are being actively commanded each frame
    commanded_joints: List[int] = []

    # Camera interaction state
    last_mouse_xy: Tuple[int, int] = (0, 0)
    left_down = False

    # Control speeds for continuous key hold
    vel = 0.8  # rad/s
    # Persistent map of pressed keys to support continuous control across frames
    key_down: Dict[int, bool] = {}

    help_text()

    try:
        while True:
            if not p.isConnected():
                print("Physics server disconnected; exiting loop.")
                break
            # Keyboard events
            events = p.getKeyboardEvents()
            digit_map = {ord("1"): 0, ord("2"): 1, ord("3"): 2, ord("4"): 3, ord("5"): 4, ord("6"): 5}
            alt_reverse = {ord("!"): 0, ord("@"): 1, ord("#"): 2, ord("$"): 3, ord("%"): 4, ord("^"): 5}

            # Modifier state: prefer SHIFT to choose reverse direction
            shift_code = getattr(p, "B3G_SHIFT", 0)
            shift_down = shift_code in events and (events[shift_code] & p.KEY_IS_DOWN)

            # Update persistent key_down map and track active digit keys
            active_keys: Dict[int, int] = {}
            commanded_joints = []

            for code, state in events.items():
                # Maintain key_down state
                if state & p.KEY_IS_DOWN:
                    key_down[code] = True
                if state & p.KEY_WAS_RELEASEED if hasattr(p, 'KEY_WAS_RELEASEED') else False:
                    # Some builds may use KEY_WAS_RELEASEED; fall back to KEY_WAS_RELEASED below
                    key_down[code] = False
                if state & p.KEY_WAS_RELEASED:
                    key_down[code] = False
                # Randomize target with 'r' or 'R'
                if state & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                    if code in (ord('r'), ord('R')):
                        tpos = random_target(bounds)
                        p.resetBasePositionAndOrientation(target_id, tpos, [0, 0, 0, 1])

                # Handle digits 1..6 with shift-based reverse
                # Continuous control based on persistent key_down
                # Handle digits 1..6 with optional shift-based reverse
                for kc, i in digit_map.items():
                    jidx = joint_indices[i]
                    if jidx is None:
                        continue
                    rev_pressed = key_down.get(list(alt_reverse.keys())[i], False)
                    fwd_pressed = key_down.get(kc, False)
                    if fwd_pressed or rev_pressed:
                        direction = -1 if (rev_pressed or shift_down) else 1
                        p.setJointMotorControl2(robot_id, jidx, p.VELOCITY_CONTROL, targetVelocity=direction * vel, force=2.0)
                        commanded_joints.append(jidx)
                    # If the specific digit was just released this frame, snap to current position
                    if events.get(kc, 0) & p.KEY_WAS_RELEASED:
                        pos = p.getJointState(robot_id, jidx)[0]
                        p.setJointMotorControl2(robot_id, jidx, p.POSITION_CONTROL, targetPosition=pos, force=2.0)

                # Alternative reverse via shifted symbols ! @ # $ % ^
                # Also stop on alt-reverse release explicitly
                for kc, i in alt_reverse.items():
                    jidx = joint_indices[i]
                    if jidx is None:
                        continue
                    if events.get(kc, 0) & p.KEY_WAS_RELEASED:
                        pos = p.getJointState(robot_id, jidx)[0]
                        p.setJointMotorControl2(robot_id, jidx, p.POSITION_CONTROL, targetPosition=pos, force=2.0)

            # Hold position for any joints not being actively commanded to avoid coasting
            for i, jidx in enumerate(joint_indices):
                if jidx is None or jidx in commanded_joints:
                    continue
                pos = p.getJointState(robot_id, jidx)[0]
                p.setJointMotorControl2(robot_id, jidx, p.POSITION_CONTROL, targetPosition=pos, force=2.0)

            # Update target from sliders and axis markers
            tpos = read_target(slider_ids)
            p.resetBasePositionAndOrientation(target_id, tpos, [0, 0, 0, 1])
            L = 0.06
            # Update axis lines using replaceItemUniqueId
            p.addUserDebugLine([tpos[0] - L, tpos[1], tpos[2]], [tpos[0] + L, tpos[1], tpos[2]], [1, 0, 0], lineWidth=2, replaceItemUniqueId=axis_ids[0])
            p.addUserDebugLine([tpos[0], tpos[1] - L, tpos[2]], [tpos[0], tpos[1] + L, tpos[2]], [0, 1, 0], lineWidth=2, replaceItemUniqueId=axis_ids[1])
            p.addUserDebugLine([tpos[0], tpos[1], tpos[2] - L], [tpos[0], tpos[1], tpos[2] + L], [0, 0, 1], lineWidth=2, replaceItemUniqueId=axis_ids[2])

            # Randomize target via GUI button (press -> resets to 0)
            try:
                rv = p.readUserDebugParameter(rand_id)
                if rv > 0.5:
                    tpos = random_target(bounds)
                    p.resetBasePositionAndOrientation(target_id, tpos, [0, 0, 0, 1])
                    # reset button state
                    if hasattr(p, "resetUserDebugParameter"):
                        p.resetUserDebugParameter(rand_id, 0)
            except Exception:
                pass

            # Mouse events for camera
            cam = p.getDebugVisualizerCamera()
            dist, yaw, pitch, target = cam[10], cam[8], cam[9], list(cam[11])
            for me in p.getMouseEvents():
                # Try to infer button and movement without relying on constants
                mouseX = me[3] if len(me) > 4 else last_mouse_xy[0]
                mouseY = me[4] if len(me) > 4 else last_mouse_xy[1]
                # Button state update
                if len(me) > 2 and me[1] == getattr(p, "B3G_LEFT_MOUSE", -1):
                    left_down = (me[2] == p.KEY_IS_DOWN)
                # Movement handling
                if left_down and (mouseX != last_mouse_xy[0] or mouseY != last_mouse_xy[1]):
                    dx = mouseX - last_mouse_xy[0]
                    dy = mouseY - last_mouse_xy[1]
                    mod = p.getKeyboardEvents()
                    shift = getattr(p, "B3G_SHIFT", None) in mod and (mod[getattr(p, "B3G_SHIFT", 0)] & p.KEY_IS_DOWN)
                    ctrl = getattr(p, "B3G_CONTROL", None) in mod and (mod[getattr(p, "B3G_CONTROL", 0)] & p.KEY_IS_DOWN)
                    if shift:  # rotate
                        yaw += dx * 0.2
                        pitch = max(-89.0, min(89.0, pitch - dy * 0.2))
                    elif ctrl:  # pan
                        target[0] -= dx * 0.001 * dist
                        target[1] += dy * 0.001 * dist
                last_mouse_xy = (mouseX, mouseY)

            p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    finally:
        try:
            p.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
