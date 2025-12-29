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
    # Performance tweaks: disable heavy previews and shadows
    if hasattr(p, 'COV_ENABLE_RGB_BUFFER_PREVIEW'):
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    if hasattr(p, 'COV_ENABLE_DEPTH_BUFFER_PREVIEW'):
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    if hasattr(p, 'COV_ENABLE_SEGMENTATION_MARK_PREVIEW'):
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    if hasattr(p, 'COV_ENABLE_SHADOWS'):
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
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
        # Level the base close to the world plane
        robot_id = p.loadURDF(urdf, basePosition=[0, 0, -0.03], useFixedBase=True, flags=flags)
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
            # Enable position control initially at current pose
            cur = p.getJointState(robot_id, j)[0]
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=cur, force=3.0)
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
    radius = 0.10
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    target_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[0, 0, 0.15])

    # Add axis debug lines centered at target for visibility
    L = 0.06
    center = [0, 0, 0.15]
    x_line = p.addUserDebugLine([center[0] - L, center[1], center[2]], [center[0] + L, center[1], center[2]], [1, 0, 0], lineWidth=2)
    y_line = p.addUserDebugLine([center[0], center[1] - L, center[2]], [center[0], center[1] + L, center[2]], [0, 1, 0], lineWidth=2)
    z_line = p.addUserDebugLine([center[0], center[1], center[2] - L], [center[0], center[1], center[2] + L], [0, 0, 1], lineWidth=2)

    return (x_id, y_id, z_id), rand_id, target_id, (x_line, y_line, z_line), bounds


def make_joint_sliders(joint_names: List[str], joint_indices: List[int], limits: Dict[int, Tuple[float, float]], robot_id: int) -> Dict[int, int]:
    """Create 6 joint sliders (1..6) mapped to joint indices; return {joint_index: slider_id}."""
    slider_ids: Dict[int, int] = {}
    for i, jidx in enumerate(joint_indices):
        if jidx is None:
            continue
        lo, hi = limits[jidx]
        cur = p.getJointState(robot_id, jidx)[0]
        sid = p.addUserDebugParameter(f"Joint {joint_names[i]}", lo, hi, cur)
        slider_ids[jidx] = sid
    return slider_ids


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
    print("  Use Joint 1..6 sliders to control motors")
    print("  Mouse + Shift: rotate camera (drag)")
    print("  Mouse + Ctrl: pan camera (drag)")
    print("  R: randomize target point within bounds")
    print("\nGUI: Target X/Y/Z sliders and 'Randomize Target' button.")


def main():
    robot_id, name_to_index, limits = load_world()
    # Expected joints named "1".."6" from URDF
    joint_names = [str(i) for i in range(1, 7)]
    joint_indices = [name_to_index[n] for n in joint_names]

    slider_ids, rand_id, target_id, axis_ids, bounds = make_target_ui()
    # Create joint sliders (position control via GUI)
    joint_slider_ids = make_joint_sliders(joint_names, joint_indices, limits, robot_id)

    # Camera interaction state
    last_mouse_xy: Tuple[int, int] = (0, 0)
    left_down = False

    # Control speeds for continuous key hold
    speed = 1.5  # rad/s for position target updates
    last_time = time.time()

    help_text()

    try:
        while True:
            if not p.isConnected():
                print("Physics server disconnected; exiting loop.")
                break
            # Keyboard events (only used for randomize and camera modifiers)
            events = p.getKeyboardEvents()

            # Frame dt
            now = time.time()
            dt = max(1e-3, min(0.05, now - last_time))
            last_time = now

            for code, state in events.items():
                # Randomize target with 'r' or 'R'
                if state & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                    if code in (ord('r'), ord('R')):
                        tpos = random_target(bounds)
                        p.resetBasePositionAndOrientation(target_id, tpos, [0, 0, 0, 1])

            # Apply joint slider targets each frame
            for jidx, sid in joint_slider_ids.items():
                t = p.readUserDebugParameter(sid)
                p.setJointMotorControl2(robot_id, jidx, p.POSITION_CONTROL, targetPosition=t, force=3.0)

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

            p.stepSimulation()
            # Cap CPU usage but aim for high FPS; adjust if needed
            time.sleep(1.0 / 24000.0)
    finally:
        try:
            p.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
