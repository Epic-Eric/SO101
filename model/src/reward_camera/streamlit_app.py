import streamlit as st
import cv2
import numpy as np
import os
import time
import torch
import matplotlib.pyplot as plt
import sys
import threading
import queue
import json
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add workspace root to path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from model.src.reward_camera.opencv_teacher import RedBeadTeacher, TeacherParams
from model.src.reward_camera.dataset import RedBeadDataset
from model.src.reward_camera.train import train_model
from model.src.reward_camera.inference import InferenceEngine
from model.src.utils.teleop import setup_so101_with_camera, step_teleop, teardown

# Page Config
st.set_page_config(page_title="Red Bead Detector & Teleop", layout="wide")

# Session State Initialization
if 'teacher_params' not in st.session_state:
    st.session_state.teacher_params = TeacherParams()
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'model_path' not in st.session_state:
    # Use absolute path relative to workspace root
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    st.session_state.model_path = os.path.join(workspace_root, "red_bead_detector_data/best_model.pth")
if 'data_dir' not in st.session_state:
    # Use absolute path relative to workspace root
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    st.session_state.data_dir = os.path.join(workspace_root, "red_bead_detector_data")
if 'robot_connected' not in st.session_state:
    st.session_state.robot_connected = False
if 'robot' not in st.session_state:
    st.session_state.robot = None
if 'teleop' not in st.session_state:
    st.session_state.teleop = None
if 'teleop_thread' not in st.session_state:
    st.session_state.teleop_thread = None
if 'teleop_queue' not in st.session_state:
    st.session_state.teleop_queue = queue.Queue(maxsize=1)
if 'teleop_error_queue' not in st.session_state:
    st.session_state.teleop_error_queue = queue.Queue()
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()

# Ensure data dir exists
os.makedirs(st.session_state.data_dir, exist_ok=True)

# --- Teleop Thread Function ---
def teleop_worker(robot, teleop, q, stop_event, error_q):
    """Runs teleop loop at high frequency."""
    print("Teleop thread started.")
    consecutive_errors = 0
    max_errors = 20  # Be lax: allow some transient errors

    while not stop_event.is_set():
        try:
            # Step teleop (Action -> Send -> Obs)
            data = step_teleop(robot, teleop)
            consecutive_errors = 0  # Reset on success
            
            # Put latest data in queue (overwrite if full)
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put(data)
            
            # Sleep to maintain ~100Hz or whatever is stable
            time.sleep(0.005) 
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors >= max_errors:
                msg = f"Teleop thread failed after {max_errors} attempts. Last error: {e}"
                print(msg)
                error_q.put(msg)
                break
            # Short sleep to allow recovery
            time.sleep(0.05)
            
    print("Teleop thread stopped.")
    # Emergency teardown if we crashed
    if consecutive_errors >= max_errors:
        print("Emergency teardown in thread...")
        try:
            teardown(robot, teleop)
        except Exception as e:
            print(f"Teardown failed: {e}")

# --- Sidebar: Controls ---
st.sidebar.title("Control Panel")

mode = st.sidebar.radio("Mode", ["Live Inference & Calibration", "Training", "Dataset View"])

st.sidebar.header("Teleoperation Setup")
# Load defaults from env
default_follower_port = os.getenv("FOLLOWER_PORT", "/dev/tty.usbmodem575E0031341")
default_leader_port = os.getenv("LEADER_PORT", "/dev/tty.usbmodem575E0032081")
default_camera_idx = int(os.getenv("REWARD_CAMERA_INDEX", 0))

follower_port = st.sidebar.text_input("Follower Port", default_follower_port)
leader_port = st.sidebar.text_input("Leader Port", default_leader_port)
camera_idx = st.sidebar.number_input("Camera Index", 0, 10, default_camera_idx)

# Connection Status Indicator
status_placeholder = st.sidebar.empty()

def update_status():
    status_color = "green" if st.session_state.robot_connected else "red"
    status_text = "Connected" if st.session_state.robot_connected else "Disconnected"
    status_placeholder.markdown(f"**Status:** :{status_color}[{status_text}]")

update_status()

if st.sidebar.button("Connect / Disconnect Robot"):
    if st.session_state.robot_connected:
        # Disconnect
        st.session_state.stop_event.set()
        if st.session_state.teleop_thread and st.session_state.teleop_thread.is_alive():
            st.session_state.teleop_thread.join()
            
        try:
            teardown(st.session_state.robot, st.session_state.teleop)
        except Exception as e:
            st.sidebar.error(f"Error disconnecting: {e}")
        finally:
            st.session_state.robot = None
            st.session_state.teleop = None
            st.session_state.robot_connected = False
            st.sidebar.success("Disconnected.")
            update_status()
    else:
        # Connect
        try:
            with st.spinner("Connecting to robot..."):
                robot, teleop = setup_so101_with_camera(
                    follower_port=follower_port,
                    leader_port=leader_port,
                    camera_index=camera_idx,
                    calibrate=True
                )
                st.session_state.robot = robot
                st.session_state.teleop = teleop
                st.session_state.robot_connected = True
            st.sidebar.success("Connected!")
            update_status()
        except Exception as e:
            st.sidebar.error(f"Connection failed: {e}")
            st.session_state.robot_connected = False
            update_status()

st.sidebar.header("OpenCV Calibration")
# HSV 1
h_low_1 = st.sidebar.slider("H Low 1", 0, 180, st.session_state.teacher_params.h_low_1)
h_high_1 = st.sidebar.slider("H High 1", 0, 180, st.session_state.teacher_params.h_high_1)
# HSV 2
h_low_2 = st.sidebar.slider("H Low 2", 0, 180, st.session_state.teacher_params.h_low_2)
h_high_2 = st.sidebar.slider("H High 2", 0, 180, st.session_state.teacher_params.h_high_2)

s_min = st.sidebar.slider("S Min", 0, 255, st.session_state.teacher_params.s_min)
v_min = st.sidebar.slider("V Min", 0, 255, st.session_state.teacher_params.v_min)

st.sidebar.subheader("Morphology & Filter")
morph_k = st.sidebar.slider("Morph Kernel", 1, 7, st.session_state.teacher_params.morph_kernel_size, step=2)
morph_iter = st.sidebar.slider("Morph Iterations", 1, 5, st.session_state.teacher_params.morph_iterations)
min_area = st.sidebar.number_input("Min Area", value=st.session_state.teacher_params.min_area)
max_area = st.sidebar.number_input("Max Area", value=st.session_state.teacher_params.max_area)
min_circ = st.sidebar.slider("Min Circularity", 0.0, 1.0, st.session_state.teacher_params.min_circularity)

# Update params
current_params = TeacherParams(
    h_low_1=h_low_1, h_high_1=h_high_1, h_low_2=h_low_2, h_high_2=h_high_2,
    s_min=s_min, v_min=v_min,
    morph_kernel_size=morph_k, morph_iterations=morph_iter,
    min_area=min_area, max_area=max_area, min_circularity=min_circ
)
st.session_state.teacher_params = current_params

if st.sidebar.button("Save Calibration Params"):
    params_path = os.path.join(st.session_state.data_dir, "teacher_params.json")
    current_params.save(params_path)
    st.sidebar.success(f"Saved to {params_path}")

if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = InferenceEngine(st.session_state.model_path)

teacher = RedBeadTeacher(current_params)
inference_engine = st.session_state.inference_engine

# --- Main Content ---

if mode == "Live Inference & Calibration":
    st.header("Live Feed & Teleop")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Inference Settings")
        inf_mode = st.radio("Inference Mode", ["opencv", "nn", "ensemble"])
        
        st.subheader("Data Collection")
        auto_label = st.checkbox("Auto-Label (Record)", value=False)
        label_mode = st.selectbox("Labeling Strategy", ["OpenCV High Conf", "Self-Supervised (Agreement)"])
        min_conf_record = st.slider("Min Confidence", 0.0, 1.0, 0.5)
        agreement_thresh = st.slider("Agreement Threshold (px)", 5, 50, 20)
        
        run_active = st.checkbox("Start Stream / Teleop", value=False)

    image_placeholder = col1.empty()
    status_placeholder = col1.empty()
    teleop_status = col1.empty()
    
    if run_active:
        # Start Teleop Thread if connected and not running
        if st.session_state.robot_connected:
            if st.session_state.teleop_thread is None or not st.session_state.teleop_thread.is_alive():
                st.session_state.stop_event.clear()
                # Clear error queue
                while not st.session_state.teleop_error_queue.empty():
                    st.session_state.teleop_error_queue.get()
                    
                st.session_state.teleop_thread = threading.Thread(
                    target=teleop_worker, 
                    args=(st.session_state.robot, st.session_state.teleop, st.session_state.teleop_queue, st.session_state.stop_event, st.session_state.teleop_error_queue),
                    daemon=True
                )
                st.session_state.teleop_thread.start()
        else:
            # Only print once or show in UI to avoid spam
            pass
        
        # If robot is not connected, try to open local camera
        cap = None
        if not st.session_state.robot_connected:
            cap = cv2.VideoCapture(camera_idx)
            if not cap.isOpened():
                st.error("Could not open camera.")
                run_active = False
        
        dataset = RedBeadDataset(st.session_state.data_dir)
        
        while run_active:
            frame_rgb = None
            
            # Check for thread errors
            if not st.session_state.teleop_error_queue.empty():
                err_msg = st.session_state.teleop_error_queue.get()
                st.error(f"Teleop Error: {err_msg}")
                st.session_state.robot_connected = False
                update_status()
                run_active = False
                break
            
            # 1. Get Frame & Teleop Step
            if st.session_state.robot_connected:
                try:
                    # Get latest data from queue
                    if not st.session_state.teleop_queue.empty():
                        teleop_data = st.session_state.teleop_queue.get()
                        
                        # Extract image from observation
                        obs = teleop_data.get("observation", {})
                        # Try common keys
                        if "front" in obs:
                            frame_rgb = obs["front"]
                        elif "observation.images.front" in obs:
                            frame_tensor = obs["observation.images.front"]
                            # Convert tensor to numpy (C, H, W) -> (H, W, C)
                            frame_rgb = frame_tensor.permute(1, 2, 0).cpu().numpy()
                            frame_rgb = (frame_rgb * 255).astype(np.uint8)
                        elif "image" in obs:
                             frame_rgb = obs["image"]
                        else:
                            # Fallback: try to find any image in values
                            for v in obs.values():
                                if isinstance(v, (np.ndarray, torch.Tensor)) and len(v.shape) == 3:
                                    if isinstance(v, torch.Tensor):
                                        frame_rgb = v.permute(1, 2, 0).cpu().numpy()
                                        frame_rgb = (frame_rgb * 255).astype(np.uint8)
                                    else:
                                        frame_rgb = v
                                    break
                        
                        teleop_status.text(f"Teleop Active | Action: {teleop_data.get('action', 'N/A')}")
                    else:
                        # No new data yet
                        time.sleep(0.01)
                        continue
                except Exception as e:
                    st.error(f"Teleop Error: {e}")
                    break
            else:
                if cap:
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        st.error("Failed to read frame.")
                        break
            
            if frame_rgb is None:
                # st.warning("No frame received.")
                time.sleep(0.01)
                continue

            # 2. OpenCV Detection
            cv_res = teacher.detect(frame_rgb)
            
            # 3. NN Inference
            nn_res = inference_engine.predict(frame_rgb)
            
            # 4. Ensemble
            final_res = inference_engine.ensemble(cv_res, nn_res, mode=inf_mode)
            
            # Visualization
            vis_img = frame_rgb.copy()
            
            # Draw OpenCV Contours (Green)
            if cv_res.get('contours'):
                cv2.drawContours(vis_img, cv_res['contours'], -1, (0, 255, 0), 2)
            
            # Draw Final Centroids (Red Cross)
            centroids = final_res.get('centroids', [])
            for pt in centroids:
                cv2.drawMarker(vis_img, pt, (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
            
            # Calculate Metrics (Distance & Radius Diff)
            dist_px = 0.0
            radius_diff = 0.0
            
            if len(centroids) == 2:
                pt1, pt2 = centroids
                dist_px = np.linalg.norm(np.array(pt1) - np.array(pt2))
                cv2.line(vis_img, pt1, pt2, (255, 255, 0), 2)
                cv2.putText(vis_img, f"Dist: {dist_px:.1f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Radius diff (only available from OpenCV for now)
                radii = cv_res.get('radii', [])
                if len(radii) >= 2:
                    radius_diff = abs(radii[0] - radii[1])
                    cv2.putText(vis_img, f"R Diff: {radius_diff:.1f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(vis_img, f"Conf: {final_res.get('cv_confidence', 0):.2f} | {final_res.get('nn_confidence', 0):.2f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Overlay Heatmap if NN is active
            if inf_mode in ['nn', 'ensemble'] and nn_res['heatmap'] is not None:
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * nn_res['heatmap']), cv2.COLORMAP_JET)
                vis_img = cv2.addWeighted(vis_img, 0.7, heatmap_colored, 0.3, 0)
            
            # Auto-Labeling Logic
            should_record = False
            record_centroids = []
            record_reason = "Auto-Label is OFF"

            if auto_label:
                if label_mode == "OpenCV High Conf":
                    if len(cv_res.get('centroids', [])) != 2:
                        record_reason = f"Waiting: need 2 beads (have {len(cv_res.get('centroids', []))})"
                    elif float(cv_res.get('confidence', 0.0)) <= float(min_conf_record):
                        record_reason = f"Waiting: OpenCV conf {float(cv_res.get('confidence', 0.0)):.2f} <= {float(min_conf_record):.2f}"
                    else:
                        should_record = True
                        record_centroids = cv_res.get('centroids', [])
                        record_reason = "Recording (OpenCV High Conf)"

                elif label_mode == "Self-Supervised (Agreement)":
                    cv_centroids = cv_res.get('centroids', [])
                    nn_centroids = nn_res.get('centroids', [])
                    if len(cv_centroids) != 2 or len(nn_centroids) != 2:
                        record_reason = f"Waiting: need 2+2 beads (cv={len(cv_centroids)}, nn={len(nn_centroids)})"
                    else:
                        c1, c2 = cv_centroids
                        n1, n2 = nn_centroids

                        d1 = np.linalg.norm(np.array(c1) - np.array(n1)) + np.linalg.norm(np.array(c2) - np.array(n2))
                        d2 = np.linalg.norm(np.array(c1) - np.array(n2)) + np.linalg.norm(np.array(c2) - np.array(n1))
                        if min(d1, d2) < agreement_thresh * 2:
                            should_record = True
                            record_centroids = final_res.get('centroids', [])
                            record_reason = "Recording (Agreement)"
                        else:
                            record_reason = f"Waiting: disagreement {min(d1, d2):.1f}px > {agreement_thresh * 2}px"

            if should_record and record_centroids:
                try:
                    sample_id = dataset.add_sample(frame_rgb, record_centroids, float(cv_res.get('confidence', 0.0)), vars(current_params))
                    status_placeholder.text(f"Recorded {sample_id} | Total: {len(dataset)} | Dir: {dataset.data_dir}")
                except Exception as e:
                    status_placeholder.text(f"Record FAILED: {e} | Dir: {dataset.data_dir}")
            else:
                status_placeholder.text(f"Not recording: {record_reason} | Dir: {dataset.data_dir}")
            
            image_placeholder.image(vis_img, channels="RGB")
            
            # Streamlit loop trick
            time.sleep(0.01) 
            
        if cap:
            cap.release()
        
        # Stop thread when unchecked
        if st.session_state.robot_connected:
            st.session_state.stop_event.set()
            if st.session_state.teleop_thread:
                st.session_state.teleop_thread.join()

elif mode == "Training":
    st.header("Model Training")
    
    epochs = st.number_input("Epochs", 1, 100, 10)
    batch_size = st.number_input("Batch Size", 1, 64, 8)
    lr = st.number_input("Learning Rate", 1e-5, 1e-1, 1e-3, format="%.5f")

    device_options = ["auto", "cpu"]
    if torch.backends.mps.is_available():
        device_options.insert(1, "mps")
    if torch.cuda.is_available():
        device_options.insert(1, "cuda")
    device_choice = st.selectbox("Device", device_options, index=0)

    epoch_progress_bar = st.progress(0)
    batch_progress_bar = st.progress(0)
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if st.button("Start Training"):
        train_losses = []
        val_losses = []
        last_batch: Dict[str, Any] = {"evt": None}

        def on_progress(evt: dict):
            if evt.get("event") == "start":
                status_placeholder.info(f"Training started on device: {evt.get('device')}")
                epoch_progress_bar.progress(0)
                batch_progress_bar.progress(0)
            elif evt.get("event") == "batch_end":
                last_batch["evt"] = evt
                e = int(evt.get("epoch", 0))
                E = int(evt.get("epochs", 1))
                b = int(evt.get("batch", 0))
                B = int(evt.get("batches", 1))
                bl = float(evt.get("batch_loss", 0.0))
                frac = min(max(b / max(B, 1), 0.0), 1.0)
                batch_progress_bar.progress(frac)
                metrics_placeholder.markdown(
                    f"**Epoch {e}/{E}**  \n"
                    f"Batch: `{b}/{B}`  \n"
                    f"Batch loss: `{bl:.6f}`  \n"
                    f"Device: `{evt.get('device')}`"
                )
            elif evt.get("event") == "epoch_end":
                e = int(evt.get("epoch", 0))
                E = int(evt.get("epochs", 1))
                tl = float(evt.get("train_loss", 0.0))
                vl = float(evt.get("val_loss", 0.0))
                train_losses.append(tl)
                val_losses.append(vl)

                frac = min(max(e / max(E, 1), 0.0), 1.0)
                epoch_progress_bar.progress(frac)
                batch_progress_bar.progress(0)

                # Keep batch info if we have it
                batch_line = ""
                last_evt = last_batch.get("evt")
                if last_evt is not None and int(last_evt.get("epoch", -1)) == e:
                    b = int(last_evt.get("batch", 0))
                    B = int(last_evt.get("batches", 1))
                    bl = float(last_evt.get("batch_loss", 0.0))
                    batch_line = (
                        f"Batch: `{b}/{B}`  \n"
                        f"Batch loss: `{bl:.6f}`  \n"
                    )

                metrics_placeholder.markdown(
                    f"**Epoch {e}/{E}**  \n"
                    f"{batch_line}"
                    f"Train loss: `{tl:.6f}`  \n"
                    f"Val loss: `{vl:.6f}`  \n"
                    f"Device: `{evt.get('device')}`"
                )
                chart_placeholder.line_chart({"train": train_losses, "val": val_losses})
            elif evt.get("event") == "end":
                status_placeholder.success(f"Training finished on device: {evt.get('device')}")

        with st.spinner("Training..."):
            result = train_model(
                st.session_state.data_dir,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(lr),
                device_name=device_choice,
                progress_callback=on_progress,
            )

        if result and result.get("history"):
            st.success(f"Training Complete! Saved best model to: {result.get('model_path')}")
        else:
            st.warning("Training failed or no data.")

elif mode == "Dataset View":
    st.header("Dataset Gallery")
    dataset = RedBeadDataset(st.session_state.data_dir)
    st.write(f"Total Samples: {len(dataset)}")

    @st.dialog("Sample Viewer")
    def show_sample_dialog(sample_id: str):
        img_path = os.path.join(dataset.data_dir, 'images', f"{sample_id}.png")
        heatmap_path = os.path.join(dataset.data_dir, 'heatmaps', f"{sample_id}.npy")

        st.markdown(f"**{sample_id}**")

        img_rgb = None
        if os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                st.warning("Could not read image file.")
        else:
            st.warning("Image file missing.")

        hm = None
        if os.path.exists(heatmap_path):
            try:
                hm = np.load(heatmap_path)
            except Exception as e:
                st.warning(f"Could not load heatmap: {e}")
        else:
            st.warning("Heatmap file missing.")

        tab1, tab2 = st.tabs(["Side-by-side", "Overlay"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                if img_rgb is not None:
                    st.image(img_rgb, caption="Image", width='stretch')
            with c2:
                if hm is not None:
                    st.image(hm, caption="Label Heatmap", clamp=True, width='stretch')

        with tab2:
            if img_rgb is not None and hm is not None:
                try:
                    hm_norm = hm
                    if hm_norm.dtype != np.uint8:
                        hm_norm = (255.0 * np.clip(hm_norm, 0.0, 1.0)).astype(np.uint8)
                    if hm_norm.ndim == 3:
                        hm_norm = hm_norm.squeeze()
                    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img_rgb, 0.7, hm_color, 0.3, 0)
                    st.image(overlay, caption="Overlay", width='stretch')
                except Exception as e:
                    st.warning(f"Could not render overlay: {e}")
            else:
                st.info("Need both image and heatmap to render overlay.")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Refresh Dataset"):
            dataset.refresh_index()
            st.rerun()
    with col2:
        if st.button("Delete All Samples"):
            if st.checkbox("Confirm Delete All"):
                dataset.delete_all_samples()
                st.success("All samples deleted!")
                st.rerun()
    with col3:
        selected_to_delete = st.multiselect("Select samples to delete", dataset.samples)
        if st.button("Delete Selected") and selected_to_delete:
            for sample_id in selected_to_delete:
                dataset.delete_sample(sample_id)
            st.success(f"Deleted {len(selected_to_delete)} samples!")
            st.rerun()

    if len(dataset) > 0:
        # Gallery view
        cols = st.columns(4)  # 4 images per row
        for i, sample_id in enumerate(dataset.samples):
            with cols[i % 4]:
                img_path = os.path.join(dataset.data_dir, 'images', f"{sample_id}.png")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img, caption=f"Sample {i+1}: {sample_id}", width='stretch')
                    else:
                        st.warning(f"Could not read {sample_id} image")

                    if st.button("View", key=f"view_{sample_id}"):
                        show_sample_dialog(sample_id)
                    
                    # Load metadata for details
                    metadata_path = os.path.join(dataset.data_dir, 'metadata', f"{sample_id}.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        centroids = metadata.get('centroids', [])
                        confidence = metadata.get('confidence', 0)
                        st.write(f"Centroids: {centroids}")
                        st.write(f"Confidence: {confidence:.2f}")
                else:
                    st.write(f"Missing image for {sample_id}")
    else:
        st.write("No samples in dataset.")

