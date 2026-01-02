import torch
import numpy as np
import cv2
from typing import Tuple, Optional, Dict
from .bead_model import RedBeadAutoEncoder

class Smoother:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value: Optional[np.ndarray] = None

    def update(self, measurement: Tuple[int, int]) -> Tuple[int, int]:
        meas_arr = np.array(measurement, dtype=np.float32)
        if self.value is None:
            self.value = meas_arr
        else:
            self.value = self.alpha * meas_arr + (1 - self.alpha) * self.value
        return tuple(self.value.astype(int))

class InferenceEngine:
    def __init__(self, model_path: str, device_name: str = 'auto'):
        if device_name == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_name)
            
        self.model = RedBeadAutoEncoder().to(self.device)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model_loaded = True
            except FileNotFoundError:
                print(f"Warning: Model not found at {model_path}. Inference will be disabled until a model is trained.")
                self.model_loaded = False
        else:
            self.model_loaded = False
            
        self.smoother = Smoother(alpha=0.5)

    def predict(self, image_rgb: np.ndarray) -> Dict:
        if not self.model_loaded:
            return {"heatmap": None, "centroid": None, "confidence": 0.0}
            
        # Preprocess
        img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            heatmap = output.squeeze().cpu().numpy()
            
        # Post-process heatmap to find centroids (up to 2)
        centroids = []
        heatmap_copy = heatmap.copy()
        
        # Find first peak
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_copy)
        if max_val > 0.1:
            centroids.append(max_loc)
            # Mask out the first peak to find the second
            cv2.circle(heatmap_copy, max_loc, 20, 0, -1) # Mask radius 20
            
            # Find second peak
            min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(heatmap_copy)
            if max_val2 > 0.1:
                centroids.append(max_loc2)
        
        # Basic confidence: peak value of first
        confidence = float(max_val)
        
        return {
            "heatmap": heatmap,
            "centroids": centroids,
            "confidence": confidence
        }

    def ensemble(self, opencv_res: Dict, nn_res: Dict, mode: str = 'ensemble') -> Dict:
        """
        Combines OpenCV and NN results. Returns list of centroids.
        """
        final_centroids = []
        
        cv_centroids = opencv_res.get('centroids', [])
        nn_centroids = nn_res.get('centroids', [])
        
        if mode == 'opencv':
            final_centroids = cv_centroids
        elif mode == 'nn':
            final_centroids = nn_centroids
        else: # Ensemble
            # If both agree on count and positions, average them.
            # This is a simplified matching logic.
            if len(cv_centroids) == len(nn_centroids) and len(cv_centroids) > 0:
                # Try to match points
                matched_nn = [False] * len(nn_centroids)
                temp_centroids = []
                
                for cv_pt in cv_centroids:
                    best_dist = float('inf')
                    best_idx = -1
                    
                    for i, nn_pt in enumerate(nn_centroids):
                        if matched_nn[i]: continue
                        dist = np.linalg.norm(np.array(cv_pt) - np.array(nn_pt))
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i
                    
                    if best_idx != -1 and best_dist < 50:
                        # Match found, average them
                        nn_pt = nn_centroids[best_idx]
                        avg_pt = tuple(np.mean([cv_pt, nn_pt], axis=0).astype(int))
                        temp_centroids.append(avg_pt)
                        matched_nn[best_idx] = True
                    else:
                        # No match, trust OpenCV if high confidence? Or just keep OpenCV
                        temp_centroids.append(cv_pt)
                
                final_centroids = temp_centroids
            else:
                # Disagreement on count, fallback to OpenCV usually more stable for "ground truth" logic
                # or NN if trained well. Let's default to OpenCV for now as teacher.
                final_centroids = cv_centroids
                
        return {
            "centroids": final_centroids,
            "cv_confidence": opencv_res.get('confidence', 0),
            "nn_confidence": nn_res.get('confidence', 0)
        }
