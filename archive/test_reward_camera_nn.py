import streamlit as st
import cv2
import numpy as np
import json
import time
import os
import torch
from datetime import datetime
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# --- Configuration & Constants ---
DATA_DIR = "data/captured_images"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
JSONL_PATH = os.path.join(DATA_DIR, "actions.jsonl")

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Helper Classes ---

class NeuralTracker:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_id = "google/owlvit-base-patch32"
        
        with st.spinner(f"Loading Neural Network ({self.model_id})..."):
            self.processor = OwlViTProcessor.from_pretrained(self.model_id)
            self.model = OwlViTForObjectDetection.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
        
        self.text_queries = ["a red bead"]
        self.last_centroid = None
        self.history = []
        self.max_history = 50

    def detect(self, frame, confidence_threshold=0.1):
        """
        Detect the red bead using OWL-ViT.
        """
        # Convert to PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Prepare inputs
        inputs = self.processor(text=self.text_queries, images=image, return_tensors="pt").to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs, threshold=confidence_threshold, target_sizes=target_sizes)[0]
        
        best_centroid = None
        best_score = 0
        best_box = None
        
        # Find best detection
        # We prioritize high confidence
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy() # We only have one label (0)
        
        candidates = []
        
        for box, score in zip(boxes, scores):
            # box is [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = box
            
            # Calculate centroid
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)
            
            # Calculate radius (approximate)
            w = xmax - xmin
            h = ymax - ymin
            radius = (w + h) / 4
            
            candidates.append(((cx, cy), radius, score, box))

        # Heuristic: Choose highest score, or closest to last if score is similar
        if candidates:
            # Sort by score descending
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Take the top one
            best_centroid, best_radius, best_score, best_box = candidates[0]
            
            self.last_centroid = best_centroid
            self.history.append(best_centroid)
            if len(self.history) > self.max_history:
                self.history.pop(0)
                
            # Create a mask for visualization (just a circle on black)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, best_centroid, int(best_radius), 255, -1)
            
            return mask, best_centroid, best_radius, best_score
            
        return np.zeros(frame.shape[:2], dtype=np.uint8), None, None, 0.0

# --- Streamlit App ---

st.set_page_config(page_title="Neural Bead Tracker", layout="wide")

# Initialize Session State
if 'tracker' not in st.session_state:
    st.session_state.tracker = NeuralTracker()
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# Sidebar
st.sidebar.title("Neural Settings")
camera_id = st.sidebar.number_input("Camera ID", min_value=0, value=0, step=1)
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.1)
text_prompt = st.sidebar.text_input("Text Prompt", "a red bead")

if text_prompt != st.session_state.tracker.text_queries[0]:
    st.session_state.tracker.text_queries = [text_prompt]

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
st.title("Neural Zero-Shot Detection (OWL-ViT)")

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
    stop_button = st.button("Stop App")
    
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame")
            break
            
        # Detection
        mask, centroid, radius, score = st.session_state.tracker.detect(frame, confidence)
        
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
            cv2.putText(vis_frame, f"{score:.2f}", (centroid[0]+10, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            telemetry_text.json({
                "x": centroid[0],
                "y": centroid[1],
                "radius": radius,
                "confidence": float(score),
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
                    "radius": radius if centroid else None,
                    "confidence": float(score) if score else None
                }
            }
            
            with open(JSONL_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
            
            st.session_state.frame_count += 1

        # Display
        vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        st_display.image(vis_frame_rgb, channels="RGB", use_column_width=True)
        mask_display.image(mask, clamp=True, channels="GRAY", use_column_width=True)

        # No sleep needed as NN is slow enough
        
    cap.release()
