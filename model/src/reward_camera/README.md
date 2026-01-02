# Red Bead Detector & Teleop

Self-supervised vision pipeline for red bead localization, integrated with SO101 Teleoperation.

## Structure
- `opencv_teacher.py`: HSV-based detector and teacher.
- `model.py`: CNN Autoencoder.
- `dataset.py`: Data management.
- `train.py`: Training loop.
- `inference.py`: Inference and ensemble logic.
- `streamlit_app.py`: Main UI with Teleop.

## Usage

1. **Install Dependencies**
   Ensure you have `torch`, `opencv-python`, `streamlit`, `matplotlib`, `numpy`, `lerobot` installed.

2. **Run the App**
   From the root directory (`so101`):
   ```bash
   streamlit run model/src/reward_camera/streamlit_app.py
   ```

3. **Workflow**
   - **Teleop Setup**: Enter Follower and Leader ports in the sidebar and click "Connect".
   - **Calibration**: Go to "Live Inference & Calibration". Adjust HSV sliders.
   - **Data Collection**: Check "Auto-Label". Use the Leader arm to move the Follower arm. The system records frames where OpenCV confidence is high.
   - **Training**: Switch to "Training" mode.
   - **Inference**: Switch back to "Live Inference". Change "Inference Mode" to "nn" or "ensemble".
