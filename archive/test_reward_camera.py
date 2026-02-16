import streamlit as st
import cv2
import numpy as np
import json
import time
import os
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# --- Configuration & Constants ---
DATA_DIR = "data/captured_images"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
JSONL_PATH = os.path.join(DATA_DIR, "actions.jsonl")

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Helper Classes ---

class RedBeadTracker:
    def __init__(self):
        # Default Red HSV ranges (OpenCV Hue is 0-179)
        # Red is usually 0-10 and 170-180
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])
        
        self.last_centroid = None
        self.last_radius = None
        self.history = []
        self.max_history = 50

    def calibrate(self, frame, x, y, tolerance=10, min_saturation=100, min_value=50):
        """
        Calibrate HSV range based on a clicked point in the frame.
        """
        if frame is None:
            return False

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get color at click
        # Ensure coordinates are within bounds
        h, w, _ = hsv.shape
        if 0 <= y < h and 0 <= x < w:
            pixel_color = hsv[y, x]
            hue = int(pixel_color[0])
            sat = int(pixel_color[1])
            val = int(pixel_color[2])
            
            # Define new ranges with tolerance
            # Handle Hue wrapping
            lower_h = (hue - tolerance) % 180
            upper_h = (hue + tolerance) % 180
            
            # Saturation and Value usually need to be high for colored objects, 
            # but we can use the clicked value with some wide tolerance or keep them fixed/wide.
            # Let's use a tolerance for S and V as well, but clamp them.
            s_tol = 30 # Stricter tolerance
            v_tol = 50
            
            lower_s = max(min_saturation, sat - s_tol) # Respect global minimum
            upper_s = min(255, sat + s_tol)
            lower_v = max(min_value, val - v_tol)
            upper_v = min(255, val + v_tol)

            # If the range wraps around 180, we need two ranges
            if lower_h > upper_h:
                self.lower_red1 = np.array([0, lower_s, lower_v])
                self.upper_red1 = np.array([upper_h, upper_s, upper_v])
                self.lower_red2 = np.array([lower_h, lower_s, lower_v])
                self.upper_red2 = np.array([180, upper_s, upper_v])
            else:
                self.lower_red1 = np.array([lower_h, lower_s, lower_v])
                self.upper_red1 = np.array([upper_h, upper_s, upper_v])
                # Disable second range if not needed (set to impossible values or same)
                self.lower_red2 = np.array([181, 0, 0]) 
                self.upper_red2 = np.array([180, 0, 0])

            # Reset history on calibration
            self.history = []
            self.last_centroid = (x, y)
            self.last_radius = None # Reset radius as we don't know it from a click
            return True
        return False

    def detect(self, frame):
        """
        Detect the red bead in the frame.
        Returns:
            mask: Binary mask of detection
            centroid: (x, y) tuple or None
            radius: float or None
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_centroid = None
        best_radius = 0
        
        candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50: # Minimum area filter
                continue
            
            # Circularity check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Filter out non-circular objects (perfect circle is 1.0)
            if circularity < 0.7: 
                continue
            
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                candidates.append((center, radius))
        
        # Heuristic: Choose the one closest to the last centroid
        if candidates:
            if self.last_centroid is not None:
                # Find closest
                def distance(p1, p2):
                    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                
                # Combined score: Distance + Radius Consistency
                # We want to minimize score.
                # If radius is consistent, we tolerate larger distances.
                # If distance is small, we tolerate radius changes (though radius shouldn't change much).
                
                def score(candidate):
                    cand_centroid, cand_radius = candidate
                    dist = distance(cand_centroid, self.last_centroid)
                    
                    radius_diff = 0
                    if self.last_radius is not None:
                        radius_diff = abs(cand_radius - self.last_radius)
                    
                    # Weighting: 
                    # 1 pixel of radius difference is roughly equivalent to K pixels of movement distance penalty.
                    # If we set K=10, then a 5 pixel radius change is like 50 pixels of movement.
                    # This makes the tracker prefer objects of similar size even if they are further away.
                    radius_weight = 10.0 
                    
                    return dist + (radius_diff * radius_weight)

                best_candidate = min(candidates, key=score)
                best_centroid, best_radius = best_candidate
            else:
                # If no history, choose the largest one (by radius)
                best_candidate = max(candidates, key=lambda c: c[1])
                best_centroid, best_radius = best_candidate
                
            self.last_centroid = best_centroid
            self.last_radius = best_radius
            self.history.append(best_centroid)
            if len(self.history) > self.max_history:
                self.history.pop(0)
        else:
            # If detection lost, maybe keep last known? Or just return None.
            # For robustness, if we lose it, we return None but keep last_centroid for re-acquisition
            pass

        return mask, best_centroid, best_radius

# --- Streamlit App ---

st.set_page_config(page_title="Red Bead Tracker", layout="wide")

# Initialize Session State
if 'tracker' not in st.session_state:
    st.session_state.tracker = RedBeadTracker()
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

# Sidebar
st.sidebar.title("Settings")
camera_id = st.sidebar.number_input("Camera ID", min_value=0, value=0, step=1)
calibration_mode = st.sidebar.checkbox("Calibration Mode", value=False)
tolerance = st.sidebar.slider("Calibration Tolerance (Hue)", 1, 50, 5) # Reduced default tolerance
min_saturation = st.sidebar.slider("Min Saturation (Filter Skin/Background)", 0, 255, 100)
min_value = st.sidebar.slider("Min Value (Brightness)", 0, 255, 50)

st.sidebar.markdown("---")
st.sidebar.subheader("Recording")
if st.sidebar.button("Start/Stop Recording"):
    st.session_state.recording = not st.session_state.recording
    if st.session_state.recording:
        st.toast("Recording Started!")
    else:
        st.toast("Recording Stopped!")

status_text = st.sidebar.empty()
if st.session_state.recording:
    status_text.error("RECORDING IN PROGRESS")
else:
    status_text.success("Ready")

# Main Area
st.title("Red Bead Detection & Recording")

col1, col2 = st.columns([2, 1])

with col1:
    st_display = st.empty()
    
with col2:
    st.subheader("Detection Mask")
    mask_display = st.empty()
    st.subheader("Telemetry")
    telemetry_text = st.empty()

# Camera Loop
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    st.error(f"Cannot open camera {camera_id}")
else:
    # We use a placeholder for the loop
    stop_button = st.button("Stop App")
    
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame")
            break
            
        # Flip frame for mirror effect (optional, but usually good for webcam)
        # frame = cv2.flip(frame, 1)
        
        # Detection
        mask, centroid, radius = st.session_state.tracker.detect(frame)
        
        # Visualization
        vis_frame = frame.copy()
        
        # Draw history
        if len(st.session_state.tracker.history) > 1:
            for i in range(1, len(st.session_state.tracker.history)):
                cv2.line(vis_frame, st.session_state.tracker.history[i-1], st.session_state.tracker.history[i], (0, 255, 255), 2)
        
        # Draw current detection
        if centroid:
            cv2.circle(vis_frame, centroid, int(radius), (0, 255, 0), 2)
            cv2.circle(vis_frame, centroid, 5, (0, 0, 255), -1)
            telemetry_text.json({
                "x": centroid[0],
                "y": centroid[1],
                "radius": radius,
                "recording": st.session_state.recording
            })
        else:
            telemetry_text.write("No detection")

        # Recording Logic
        if st.session_state.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_filename = f"{timestamp}.jpg"
            img_path = os.path.join(IMAGES_DIR, img_filename)
            
            # Save Image
            cv2.imwrite(img_path, frame)
            
            # Save Metadata
            record = {
                "timestamp": timestamp,
                "image_path": img_path,
                "detection": {
                    "x": centroid[0] if centroid else None,
                    "y": centroid[1] if centroid else None,
                    "radius": radius if centroid else None
                }
            }
            
            with open(JSONL_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
            
            st.session_state.frame_count += 1

        # Display
        # Convert BGR to RGB for Streamlit
        vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        
        if calibration_mode:
            st_display.empty() # Clear the video placeholder
            
            with col1:
                st.info("Calibration Mode: Click on the red bead to calibrate.")
                
                # Convert to PIL for canvas
                pil_img = Image.fromarray(vis_frame_rgb)
                
                # Resize for display to fit screen
                display_width = 600
                aspect_ratio = pil_img.height / pil_img.width
                display_height = int(display_width * aspect_ratio)
                pil_img_resized = pil_img.resize((display_width, display_height))
                
                # Use a dynamic key to reset canvas after calibration
                canvas_key = f"calibration_canvas_{st.session_state.canvas_key}"
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=3,
                    stroke_color="#000",
                    background_image=pil_img_resized,
                    update_streamlit=True,
                    height=display_height,
                    width=display_width,
                    drawing_mode="point",
                    key=canvas_key,
                )
                
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if objects:
                        # Get last click
                        last_obj = objects[-1]
                        cx_disp = int(last_obj["left"])
                        cy_disp = int(last_obj["top"])
                        
                        # Scale back to original frame coordinates
                        scale_x = pil_img.width / display_width
                        scale_y = pil_img.height / display_height
                        
                        cx = int(cx_disp * scale_x)
                        cy = int(cy_disp * scale_y)
                        
                        # Perform calibration
                        if st.session_state.tracker.calibrate(frame, cx, cy, tolerance, min_saturation, min_value):
                            st.success(f"Calibrated at ({cx}, {cy})")
                            # Increment key to reset canvas on next run
                            st.session_state.canvas_key += 1
                            # Rerun to clear the point and update state
                            st.rerun()
                            
                # Break loop to wait for interaction
                break 

        else:
            # Live Mode
            st_display.image(vis_frame_rgb, channels="RGB", use_column_width=True)
            mask_display.image(mask, clamp=True, channels="GRAY", use_column_width=True)

        # Small delay to prevent maxing out CPU
        time.sleep(0.01)

    cap.release()

    # If we broke out due to calibration mode
    if calibration_mode:
        # We are now outside the loop, showing the canvas.
        # We need a way to resume or keep updating.
        # Streamlit script runs top to bottom.
        # If we are here, the script finished.
        # The canvas is displayed. If user interacts, script reruns.
        pass
