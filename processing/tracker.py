"""
Player tracking module to maintain consistent IDs across frames.
Uses IoU (Intersection over Union) matching for simple and fast tracking.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class PlayerTracker:
    """
    Simple tracker that matches players between frames using IoU.
    Maintains consistent IDs for clicked players.
    """
    
    def __init__(self, iou_threshold=0.3, max_lost_frames=5):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU to consider a match
            max_lost_frames: Number of frames to keep a lost player before removing
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 0
        self.tracked_players = {}  # id -> {bbox, lost_frames}
        self.id_mapping = {}  # current detection index -> tracked id
        
    def compute_iou(self, box1, box2):
        """Compute Intersection over Union of two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update(self, detections):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of bounding boxes [N, 4]
            
        Returns:
            id_mapping: Dictionary mapping detection index to tracked ID
            tracked_boxes: List of tracked boxes in same order as detections
        """
        if len(detections) == 0:
            # No detections, increment lost frames for all tracked players
            to_remove = []
            for player_id, player_data in self.tracked_players.items():
                player_data['lost_frames'] += 1
                if player_data['lost_frames'] > self.max_lost_frames:
                    to_remove.append(player_id)
            for player_id in to_remove:
                del self.tracked_players[player_id]
            return {}, []
        
        if len(self.tracked_players) == 0:
            # First frame, create new tracks for all detections
            for i, bbox in enumerate(detections):
                self.tracked_players[self.next_id] = {
                    'bbox': bbox,
                    'lost_frames': 0
                }
                self.id_mapping[i] = self.next_id
                self.next_id += 1
            return self.id_mapping, detections
        
        # Build cost matrix (1 - IoU) for Hungarian algorithm
        tracked_ids = list(self.tracked_players.keys())
        tracked_bboxes = [self.tracked_players[pid]['bbox'] for pid in tracked_ids]
        
        cost_matrix = np.zeros((len(tracked_ids), len(detections)))
        for t_idx, t_bbox in enumerate(tracked_bboxes):
            for d_idx, d_bbox in enumerate(detections):
                iou = self.compute_iou(t_bbox, d_bbox)
                cost_matrix[t_idx, d_idx] = 1 - iou  # We want to minimize cost (maximize IoU)
        
        # Solve assignment problem
        t_indices, d_indices = linear_sum_assignment(cost_matrix)
        
        # Update matches
        used_detections = set()
        self.id_mapping = {}
        
        for t_idx, d_idx in zip(t_indices, d_indices):
            if cost_matrix[t_idx, d_idx] < 1 - self.iou_threshold:
                # Good match
                player_id = tracked_ids[t_idx]
                self.tracked_players[player_id]['bbox'] = detections[d_idx]
                self.tracked_players[player_id]['lost_frames'] = 0
                self.id_mapping[d_idx] = player_id
                used_detections.add(d_idx)
        
        # Create new tracks for unmatched detections
        for d_idx in range(len(detections)):
            if d_idx not in used_detections:
                self.tracked_players[self.next_id] = {
                    'bbox': detections[d_idx],
                    'lost_frames': 0
                }
                self.id_mapping[d_idx] = self.next_id
                self.next_id += 1
        
        # Update lost frames for unmatched tracked players
        matched_t_ids = set([tracked_ids[t_idx] for t_idx in t_indices])
        for player_id in self.tracked_players.keys():
            if player_id not in matched_t_ids:
                self.tracked_players[player_id]['lost_frames'] += 1
        
        # Remove lost players
        to_remove = [pid for pid, data in self.tracked_players.items() 
                    if data['lost_frames'] > self.max_lost_frames]
        for pid in to_remove:
            del self.tracked_players[pid]
        
        # Prepare tracked boxes in detection order
        tracked_boxes = []
        for d_idx in range(len(detections)):
            if d_idx in self.id_mapping:
                tracked_boxes.append(self.tracked_players[self.id_mapping[d_idx]]['bbox'])
            else:
                tracked_boxes.append(detections[d_idx])  # Fallback
        
        return self.id_mapping, tracked_boxes
    
    def get_selected_ids(self, selected_indices):
        """
        Convert selected detection indices to tracked IDs.
        
        Args:
            selected_indices: Set of detection indices that are selected
            
        Returns:
            Set of tracked IDs that correspond to the selected detections
        """
        selected_ids = set()
        for det_idx in selected_indices:
            if det_idx in self.id_mapping:
                selected_ids.add(self.id_mapping[det_idx])
        return selected_ids
    
    def get_detection_indices(self, selected_ids):
        """
        Convert selected tracked IDs to current detection indices.
        
        Args:
            selected_ids: Set of tracked IDs that are selected
            
        Returns:
            Set of current detection indices for those tracked IDs
        """
        det_indices = set()
        reverse_mapping = {v: k for k, v in self.id_mapping.items()}
        for pid in selected_ids:
            if pid in reverse_mapping:
                det_indices.add(reverse_mapping[pid])
        return det_indices
