"""
Visual effects module.
Applies translucency to selected players by simple blending.
"""

import cv2
import numpy as np
from config import SELECTED_PLAYER_OPACITY
from .opacity import calculate_batch_opacity


def apply_translucency(frame, boxes, masks, selected_players, frame_shape):
    """
    Make selected players semi-transparent by blending with original frame.
    """
    if boxes is None or len(boxes) == 0:
        return frame.copy()
    
    # If no players selected, return original frame
    if len(selected_players) == 0:
        return frame.copy()
    
    h, w = frame_shape
    
    # Start with original frame
    result = frame.copy()
    
    if masks is not None and len(masks) > 0:
        # Process each selected player
        for player_idx in selected_players:
            if player_idx >= len(masks):
                continue
                
            # Get mask for this player
            mask = masks[player_idx]
            
            # Resize mask if needed
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Ensure mask is binary (0-255)
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            mask = (mask > 0).astype(np.uint8) * 255
            
            # Get the player region from original frame
            player_region = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Create a faded version of the player
            alpha = SELECTED_PLAYER_OPACITY
            faded_player = cv2.addWeighted(player_region, alpha, np.zeros_like(frame), 0, 0)
            
            # Create an inverse mask for the background
            mask_inv = cv2.bitwise_not(mask)
            
            # Get the background region from the current result
            background = cv2.bitwise_and(result, result, mask=mask_inv)
            
            # Combine: background + faded player
            result = cv2.add(background, faded_player)
    
    return result


def create_debug_frame(frame, boxes, masks, opacities, selected_players):
    """
    Create a debug frame showing what's being detected.
    """
    debug = frame.copy()
    
    if boxes is None or len(boxes) == 0:
        cv2.putText(debug, "No players detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return debug
    
    for i, (box, opacity) in enumerate(zip(boxes, opacities)):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Color based on selection
        if i in selected_players:
            color = (0, 0, 255)  # Red for selected
            # Add text to show it's selected
            cv2.putText(debug, f"SELECTED", (x1, y1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            color = (0, 255, 0)  # Green for normal
        
        # Draw bounding box
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        
        # Add ID text
        cv2.putText(debug, f"ID:{i}", (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return debug