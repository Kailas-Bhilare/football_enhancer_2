"""
Player detection module using YOLOv8 segmentation.
Handles loading the model and running inference on frames.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch


class PlayerDetector:
    """
    Detects players in video frames using YOLOv8 segmentation.
    Returns bounding boxes and segmentation masks for each player.
    """
    
    def __init__(self, model_name='yolov8m-seg.pt', classes=[0], device=None):
        """
        Initialize the YOLO detector.
        
        Args:
            model_name: YOLO model filename (e.g., 'yolov8m-seg.pt')
            classes: List of COCO class IDs to detect (0 = person)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Auto-select device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading YOLO model on {self.device}...")
        self.model = YOLO(model_name)
        
        # Move model to appropriate device
        if self.device == 'cuda':
            self.model.to('cuda')
            
        self.classes = classes
        print(f"Model loaded. Detecting classes: {classes}")
        
        # Store last inference results (for debugging/performance)
        self.last_inference_time = None
        
    def detect(self, frame):
        """
        Run detection on a single frame.
        
        Args:
            frame: numpy array (H, W, 3) in BGR format (OpenCV default)
            
        Returns:
            boxes: numpy array of shape (N, 4) with [x1, y1, x2, y2] coordinates
            masks: numpy array of shape (N, H, W) with binary masks, or None if no masks
        """
        if frame is None:
            return None, None
            
        # Run inference
        results = self.model(frame, classes=self.classes, verbose=False)[0]
        
        # Extract boxes and masks
        if results.boxes is None or len(results.boxes) == 0:
            return np.array([]), None
            
        # Get bounding boxes (xyxy format)
        boxes = results.boxes.xyxy.cpu().numpy()
        
        # Get segmentation masks if available
        masks = None
        if results.masks is not None:
            # Masks are returned as a Masks object with .data attribute
            # Shape: (N, orig_h, orig_w) after resizing
            masks = results.masks.data.cpu().numpy()
            
            # Binarize masks (values > 0.5 become 1, others 0)
            masks = (masks > 0.5).astype(np.uint8)
            
        return boxes, masks
    
    def detect_with_confidence(self, frame, confidence_threshold=0.5):
        """
        Run detection with confidence filtering.
        
        Args:
            frame: Input frame
            confidence_threshold: Minimum confidence score to keep detection
            
        Returns:
            boxes, masks, confidences
        """
        results = self.model(frame, classes=self.classes, verbose=False)[0]
        
        if results.boxes is None or len(results.boxes) == 0:
            return np.array([]), None, np.array([])
            
        # Get confidences
        confidences = results.boxes.conf.cpu().numpy()
        
        # Filter by confidence
        keep_idx = confidences >= confidence_threshold
        
        boxes = results.boxes.xyxy.cpu().numpy()[keep_idx]
        confidences = confidences[keep_idx]
        
        masks = None
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()[keep_idx]
            masks = (masks > 0.5).astype(np.uint8)
            
        return boxes, masks, confidences
    
    def detect_resized(self, frame, target_size=(640, 640)):
        """
        Resize frame before detection for faster inference.
        Useful for high-resolution videos.
        
        Args:
            frame: Input frame
            target_size: (width, height) for inference
            
        Returns:
            boxes (in original frame coordinates), masks (resized to original)
        """
        orig_h, orig_w = frame.shape[:2]
        
        # Resize frame
        frame_resized = cv2.resize(frame, target_size)
        
        # Run detection
        boxes_resized, masks_resized = self.detect(frame_resized)
        
        if boxes_resized is None or len(boxes_resized) == 0:
            return np.array([]), None
            
        # Scale boxes back to original coordinates
        scale_x = orig_w / target_size[0]
        scale_y = orig_h / target_size[1]
        
        boxes = boxes_resized.copy()
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        # Resize masks to original frame size
        if masks_resized is not None:
            masks = []
            for mask in masks_resized:
                mask_resized = cv2.resize(mask, (orig_w, orig_h), 
                                          interpolation=cv2.INTER_NEAREST)
                masks.append(mask_resized)
            masks = np.array(masks)
        else:
            masks = None
            
        return boxes, masks
    
    def get_model_info(self):
        """Return information about the loaded model."""
        return {
            'device': self.device,
            'model_name': self.model.model_name if hasattr(self.model, 'model_name') else 'unknown',
            'classes': self.classes
        }
