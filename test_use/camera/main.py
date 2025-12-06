import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Construct an `OpenCVCameraConfig` with your desired FPS, resolution, color mode, and rotation.
config = OpenCVCameraConfig(
    index_or_path=0,
    fps=30,
    width=1920,
    height=1080,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

# Instantiate and connect an `OpenCVCamera`, performing a warm-up read (default).
camera = OpenCVCamera(config)
camera.connect()

# Display frames with OpenCV. Press 'q' to quit.
try:
    while True:
        frame = camera.async_read(timeout_ms=200)
        # Convert RGB to BGR for OpenCV display if needed
        if config.color_mode == ColorMode.RGB:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        cv2.imshow("LeRobot Camera", frame_bgr)
        # 1ms wait; break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.disconnect()
    cv2.destroyAllWindows()