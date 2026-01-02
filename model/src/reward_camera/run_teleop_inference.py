import os
import cv2
import time
import numpy as np
import argparse
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from model.src.utils.teleop import setup_so101_with_camera, step_teleop, teardown
from model.src.reward_camera.opencv_teacher import RedBeadTeacher, TeacherParams
from model.src.reward_camera.inference import InferenceEngine
from model.src.reward_camera.dataset import RedBeadDataset

def _load_env() -> None:
    local_env = Path(__file__).with_name(".env")
    if local_env.exists():
        load_dotenv(dotenv_path=str(local_env), override=False)
    else:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(dotenv_path=env_path, override=False)

def main():
    _load_env()
    
    parser = argparse.ArgumentParser(description="Run Teleop + Red Bead Detection")
    parser.add_argument("--data-dir", default="red_bead_detector_data", help="Directory for dataset and models")
    parser.add_argument("--auto-label", action="store_true", help="Enable auto-labeling")
    parser.add_argument("--inference-mode", default="ensemble", choices=["opencv", "nn", "ensemble"])
    args = parser.parse_args()
    
    # Load Params
    params_path = os.path.join(args.data_dir, "teacher_params.json")
    teacher_params = TeacherParams.load(params_path)
    print(f"Loaded teacher params from {params_path}")
    
    # Initialize Components
    print("Initializing components...")
    teacher = RedBeadTeacher(teacher_params)
    model_path = os.path.join(args.data_dir, "best_model.pth")
    inference_engine = InferenceEngine(model_path)
    dataset = RedBeadDataset(args.data_dir)
    print("Components initialized.")
    
    # Robot Setup
    follower_port = os.environ.get("FOLLOWER_PORT")
    leader_port = os.environ.get("LEADER_PORT")
    
    if not follower_port or not leader_port:
        print("Error: FOLLOWER_PORT and LEADER_PORT must be set in .env")
        return

    print("Connecting to robot...")
    robot, teleop = setup_so101_with_camera(
        follower_port=follower_port,
        leader_port=leader_port,
        camera_index=int(os.environ.get("CAMERA_INDEX", 0)),
        calibrate=True
    )
    print("Robot connected!")
    
    try:
        while True:
            # 1. Teleop Step
            step_data = step_teleop(robot, teleop)
            
            # 2. Get Image
            obs = step_data.get("observation", {})
            frame_rgb = None
            
            if "front" in obs:
                frame_rgb = obs["front"]
            elif "observation.images.front" in obs:
                frame_tensor = obs["observation.images.front"]
                frame_rgb = frame_tensor.permute(1, 2, 0).cpu().numpy()
                frame_rgb = (frame_rgb * 255).astype(np.uint8)
            elif "image" in obs:
                frame_rgb = obs["image"]
            else:
                # Fallback search
                for v in obs.values():
                    if isinstance(v, np.ndarray) and len(v.shape) == 3:
                        frame_rgb = v
                        break
            
            if frame_rgb is None:
                time.sleep(0.001)
                continue
                
            # 3. Detection
            cv_res = teacher.detect(frame_rgb)
            nn_res = inference_engine.predict(frame_rgb)
            final_res = inference_engine.ensemble(cv_res, nn_res, mode=args.inference_mode)
            
            # 4. Visualization
            vis_img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw Contour
            if cv_res['best_contour'] is not None:
                cv2.drawContours(vis_img, [cv_res['best_contour']], -1, (0, 255, 0), 2)
                
            # Draw Centroid
            centroid = final_res['centroid']
            if centroid:
                cv2.drawMarker(vis_img, centroid, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                
            # Overlay Heatmap
            if args.inference_mode in ['nn', 'ensemble'] and nn_res['heatmap'] is not None:
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * nn_res['heatmap']), cv2.COLORMAP_JET)
                vis_img = cv2.addWeighted(vis_img, 0.7, heatmap_colored, 0.3, 0)
                
            # Info
            cv2.putText(vis_img, f"Mode: {args.inference_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_img, f"Conf: {final_res.get('cv_confidence', 0):.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if args.auto_label:
                cv2.putText(vis_img, "RECORDING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Auto-label logic (Simple OpenCV High Conf for now)
                if cv_res['confidence'] > 0.8 and cv_res['centroid'] is not None:
                    dataset.add_sample(frame_rgb, cv_res['centroid'], cv_res['confidence'], vars(teacher_params))
            
            cv2.imshow("Teleop + Red Bead Detector", vis_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                args.auto_label = not args.auto_label
                print(f"Auto-label: {args.auto_label}")
                
    except KeyboardInterrupt:
        pass
    finally:
        teardown(robot, teleop)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
