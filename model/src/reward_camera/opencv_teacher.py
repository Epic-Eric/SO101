import cv2
import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any

@dataclass
class TeacherParams:
    h_low_1: int = 0
    h_high_1: int = 10
    h_low_2: int = 170
    h_high_2: int = 180
    s_min: int = 100
    v_min: int = 100
    morph_kernel_size: int = 3
    morph_iterations: int = 2
    min_area: float = 50.0
    max_area: float = 5000.0
    min_circularity: float = 0.5
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
            
    @classmethod
    def load(cls, path: str):
        if not os.path.exists(path):
            return cls()
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
class RedBeadTeacher:
    def __init__(self, params: TeacherParams = TeacherParams()):
        self.params = params

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, v)

    def detect(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Detects the red bead in the image.
        Returns a dictionary containing detection results and debug info.
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # HSV Thresholding (handling red wrapping around 180)
        lower1 = np.array([self.params.h_low_1, self.params.s_min, self.params.v_min])
        upper1 = np.array([self.params.h_high_1, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        lower2 = np.array([self.params.h_low_2, self.params.s_min, self.params.v_min])
        upper2 = np.array([self.params.h_high_2, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphology
        kernel = np.ones((self.params.morph_kernel_size, self.params.morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.params.morph_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.params.morph_iterations)
        
        # Contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None
        max_score = -1.0
        centroid = None
        
        valid_contours = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.params.min_area <= area <= self.params.max_area):
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity < self.params.min_circularity:
                continue
            
            # Solidity check (Area / Convex Hull Area or BBox Area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity < 0.8: # Expect solid object
                    continue
            
            # Calculate score based on circularity and area
            score = circularity * np.log(area)
            
            # Estimate radius from area (assuming circle)
            radius = np.sqrt(area / np.pi)
            
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                valid_contours.append({
                    'contour': cnt,
                    'score': score,
                    'centroid': (cx, cy),
                    'radius': radius,
                    'area': area
                })
        
        # Sort by score descending
        valid_contours.sort(key=lambda x: x['score'], reverse=True)
        
        # Pick top 2
        top_contours = valid_contours[:2]
        
        centroids = [c['centroid'] for c in top_contours]
        radii = [c['radius'] for c in top_contours]
        contours_list = [c['contour'] for c in top_contours]
        scores = [c['score'] for c in top_contours]
        
        # Calculate confidence as mean score of found objects (normalized roughly)
        confidence = 0.0
        if scores:
            confidence = min(np.mean(scores) / 10.0, 1.0) # Heuristic normalization

        return {
            "centroids": centroids, # List of (x, y)
            "radii": radii,         # List of float
            "contours": contours_list,
            "confidence": confidence,
            "count": len(centroids),
            "debug_mask": mask
        }
