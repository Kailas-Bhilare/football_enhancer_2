"""
Opacity calculation module.
Maps player distance (proxied by bounding box size) to opacity values.
"""

import numpy as np
from config import *


def calculate_opacity_from_bbox(bbox, frame_area):
    """
    Calculate opacity based on bounding box area.
    Larger boxes (closer players) get higher opacity.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        frame_area: total area of the frame (height * width)
        
    Returns:
        opacity: float between OPACITY_MIN and OPACITY_MAX
    """
    # Calculate bounding box area
    x1, y1, x2, y2 = bbox[:4]  # in case bbox has extra columns (confidence etc.)
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    
    # Calculate area ratio
    area_ratio = bbox_area / frame_area
    
    # Map area ratio to opacity
    # Clamp area ratio to [min_ratio, max_ratio] range
    min_ratio = AREA_RATIO_FOR_MIN_OPACITY
    max_ratio = AREA_RATIO_FOR_MAX_OPACITY
    
    # If area ratio is below min, use min opacity
    if area_ratio <= min_ratio:
        return OPACITY_MIN
    
    # If area ratio is above max, use max opacity
    if area_ratio >= max_ratio:
        return OPACITY_MAX
    
    # Linear interpolation between min and max
    # Map [min_ratio, max_ratio] to [OPACITY_MIN, OPACITY_MAX]
    t = (area_ratio - min_ratio) / (max_ratio - min_ratio)
    opacity = OPACITY_MIN + t * (OPACITY_MAX - OPACITY_MIN)
    
    return opacity


def calculate_batch_opacity(bboxes, frame_area):
    """
    Calculate opacity for multiple bounding boxes.
    
    Args:
        bboxes: numpy array of shape (N, 4+) with bounding boxes
        frame_area: total area of the frame
        
    Returns:
        opacities: numpy array of shape (N,) with opacity values
    """
    if bboxes is None or len(bboxes) == 0:
        return np.array([])
    
    opacities = []
    for bbox in bboxes:
        opacity = calculate_opacity_from_bbox(bbox, frame_area)
        opacities.append(opacity)
    
    return np.array(opacities)


def get_distance_proxy(bbox, frame_area):
    """
    Alternative: return raw distance proxy (inverse of size).
    Useful if you want to use the value for something else.
    
    Returns:
        distance_proxy: smaller = closer, larger = farther
    """
    x1, y1, x2, y2 = bbox[:4]
    bbox_area = (x2 - x1) * (y2 - y1)
    
    # Avoid division by zero
    if bbox_area < 1:
        return float('inf')
    
    # Inverse of area ratio (normalized)
    area_ratio = bbox_area / frame_area
    return 1.0 / area_ratio if area_ratio > 0 else float('inf')


# Example of more sophisticated distance estimation using MiDaS
# (commented out - requires additional dependencies)
"""
import torch

class DepthEstimator:
    def __init__(self):
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.midas.to('cuda')
        self.midas.eval()
        
        # Transform for input
        self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
        
    def estimate_depth(self, frame):
        # Convert BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_batch = self.transform(img).to('cuda')
        
        # Inference
        with torch.no_grad():
            depth = self.midas(input_batch)
            
            # Resize to original dimensions
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
            
        return depth.cpu().numpy()
"""
