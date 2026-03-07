"""
Visualization utilities for the football enhancer.
Handles drawing overlays, instructions, and debug information.
"""

import cv2
import numpy as np


def draw_selection_overlay(frame, boxes, selected_players, color=(0, 0, 255), thickness=3):
    """
    Draw visual indicators for manually selected players.
    
    Args:
        frame: Input frame
        boxes: Array of bounding boxes [N, 4+]
        selected_players: Set of indices of selected players
        color: BGR color for the outline (default: red)
        thickness: Outline thickness
        
    Returns:
        frame with selection outlines drawn
    """
    if boxes is None or len(boxes) == 0 or not selected_players:
        return frame
    
    output = frame.copy()
    
    for idx in selected_players:
        if idx < len(boxes):
            box = boxes[idx]
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Draw a thick outline
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Add a "SELECTED" label
            label = "SELECTED"
            label_bg = (0, 0, 200)  # Dark red background
            label_fg = (255, 255, 255)  # White text
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                output,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                label_bg,
                -1
            )
            
            # Draw label text
            cv2.putText(
                output,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                label_fg,
                2
            )
    
    return output


def draw_bounding_boxes(frame, boxes, opacities=None, colors=None, thickness=2):
    """
    Draw bounding boxes around all detected players.
    
    Args:
        frame: Input frame
        boxes: Array of bounding boxes [N, 4+]
        opacities: Optional array of opacity values for color coding
        colors: Optional list of colors (one per box) or single color
        thickness: Line thickness
        
    Returns:
        frame with bounding boxes drawn
    """
    if boxes is None or len(boxes) == 0:
        return frame
    
    output = frame.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Determine color
        if colors is not None:
            if isinstance(colors, list) and i < len(colors):
                color = colors[i]
            else:
                color = colors
        elif opacities is not None and i < len(opacities):
            # Color based on opacity: blue (low) to green (high)
            opacity = opacities[i]
            blue = int(255 * (1 - opacity))
            green = int(255 * opacity)
            color = (blue, green, 0)
        else:
            color = (0, 255, 0)  # Default green
        
        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        
        # Optionally add player index
        cv2.putText(
            output,
            str(i),
            (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return output


def draw_instructions(frame, effect_enabled, selected_count):
    """
    Draw instruction overlay on the frame.
    
    Args:
        frame: Input frame
        effect_enabled: Whether the effect is currently enabled
        selected_count: Number of manually selected players
        
    Returns:
        frame with instructions drawn
    """
    output = frame.copy()
    h, w = frame.shape[:2]
    
    # Semi-transparent black bar at the bottom
    overlay = output.copy()
    bar_height = 80
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
    
    # Instructions text
    instructions = [
        f"Effect: {'ON' if effect_enabled else 'OFF'} (Press 't' to toggle)",
        f"Selected players: {selected_count} (Left-click to select, right-click to deselect)",
        "Press 'd' to clear all selections, 'q' to quit"
    ]
    
    y_offset = h - 60
    for i, text in enumerate(instructions):
        cv2.putText(
            output,
            text,
            (10, y_offset + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    # Add FPS or other info in top corner (can be updated externally)
    return output


def create_side_by_side_view(original_frame, processed_frame):
    """
    Create a side-by-side comparison of original and processed frames.
    
    Args:
        original_frame: Original unprocessed frame
        processed_frame: Frame with translucency effect applied
        
    Returns:
        Combined frame with original on left, processed on right
    """
    # Make sure frames are the same size
    if original_frame.shape != processed_frame.shape:
        # Resize processed to match original
        h, w = original_frame.shape[:2]
        processed_frame = cv2.resize(processed_frame, (w, h))
    
    # Add labels
    original_label = original_frame.copy()
    processed_label = processed_frame.copy()
    
    # Draw labels
    cv2.putText(
        original_label,
        "Original",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    cv2.putText(
        processed_label,
        "Enhanced",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
    # Combine horizontally
    combined = np.hstack((original_label, processed_label))
    
    return combined


def draw_opacity_heatmap(frame, boxes, opacities, alpha=0.3):
    """
    Create a heatmap overlay showing player opacities.
    Useful for debugging the distance-to-opacity mapping.
    
    Args:
        frame: Input frame
        boxes: Array of bounding boxes [N, 4+]
        opacities: Array of opacity values for each player
        alpha: Transparency of the heatmap overlay
        
    Returns:
        frame with heatmap overlay
    """
    if boxes is None or len(boxes) == 0 or opacities is None:
        return frame
    
    h, w = frame.shape[:2]
    
    # Create blank heatmap
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    
    for box, opacity in zip(boxes, opacities):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Ensure coordinates are within frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Color based on opacity: blue (low) to red (high)
        # Using a colormap would be better, but this is simple
        blue = int(255 * (1 - opacity))
        red = int(255 * opacity)
        color = (blue, 0, red)
        
        # Fill rectangle with color
        cv2.rectangle(heatmap, (x1, y1), (x2, y2), color, -1)
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    
    return result


def create_control_panel_overlay(frame, stats):
    """
    Create a modern-looking control panel overlay with statistics.
    
    Args:
        frame: Input frame
        stats: Dictionary containing statistics to display
            e.g., {
                'fps': 30,
                'players_detected': 22,
                'selected': 2,
                'avg_opacity': 0.65,
                'model': 'yolov8m-seg'
            }
    
    Returns:
        frame with control panel overlay
    """
    output = frame.copy()
    h, w = frame.shape[:2]
    
    # Panel dimensions
    panel_width = 220
    panel_height = 150
    margin = 10
    
    # Create semi-transparent panel in top-right corner
    overlay = output.copy()
    cv2.rectangle(
        overlay,
        (w - panel_width - margin, margin),
        (w - margin, margin + panel_height),
        (30, 30, 30),  # Dark gray
        -1
    )
    cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)
    
    # Draw panel border
    cv2.rectangle(
        output,
        (w - panel_width - margin, margin),
        (w - margin, margin + panel_height),
        (100, 100, 100),  # Light gray border
        1
    )
    
    # Panel title
    cv2.putText(
        output,
        "CONTROL PANEL",
        (w - panel_width - margin + 10, margin + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )
    
    # Statistics
    y_offset = margin + 50
    line_height = 20
    
    for key, value in stats.items():
        text = f"{key.replace('_', ' ').title()}: {value}"
        cv2.putText(
            output,
            text,
            (w - panel_width - margin + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
        y_offset += line_height
    
    return output


def draw_player_info(frame, boxes, player_ids, info_texts, colors=None):
    """
    Draw detailed information for each player.
    
    Args:
        frame: Input frame
        boxes: Array of bounding boxes
        player_ids: List/array of player identifiers
        info_texts: List of strings to display for each player
        colors: Optional list of colors for each player
        
    Returns:
        frame with player info drawn
    """
    if boxes is None or len(boxes) == 0:
        return frame
    
    output = frame.copy()
    
    for i, (box, player_id, info) in enumerate(zip(boxes, player_ids, info_texts)):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Determine color
        if colors is not None and i < len(colors):
            color = colors[i]
        else:
            color = (0, 255, 0)
        
        # Draw player ID
        id_text = f"ID: {player_id}"
        cv2.putText(
            output,
            id_text,
            (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        
        # Draw additional info
        cv2.putText(
            output,
            info,
            (x1 + 5, y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )
    
    return output


def add_timestamp(frame, timestamp_ms):
    """
    Add a timestamp to the frame.
    
    Args:
        frame: Input frame
        timestamp_ms: Timestamp in milliseconds
        
    Returns:
        frame with timestamp
    """
    output = frame.copy()
    
    # Convert milliseconds to minutes:seconds
    total_seconds = timestamp_ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    milliseconds = timestamp_ms % 1000
    
    timestamp_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    cv2.putText(
        output,
        timestamp_str,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    return output


def create_export_frame(frame, boxes, masks, selected_players, export_info):
    """
    Create a frame suitable for export/saving with metadata.
    
    Args:
        frame: Processed frame
        boxes: Bounding boxes
        masks: Segmentation masks
        selected_players: Set of selected player indices
        export_info: Dictionary with export metadata
        
    Returns:
        Frame with metadata overlay
    """
    output = frame.copy()
    h, w = frame.shape[:2]
    
    # Add metadata bar at the top
    bar_height = 40
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    
    # Format metadata text
    meta_text = f"Players: {len(boxes) if boxes is not None else 0} | Selected: {len(selected_players)} | "
    meta_text += " | ".join([f"{k}: {v}" for k, v in export_info.items()])
    
    # Wrap text if too long
    max_chars = w // 10  # Approximate
    if len(meta_text) > max_chars:
        meta_text = meta_text[:max_chars-3] + "..."
    
    cv2.putText(
        output,
        meta_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    return output


# Optional: Simple FPS counter class
class FPSCounter:
    """Simple FPS counter for performance monitoring."""
    
    def __init__(self, avg_frames=30):
        self.timestamps = []
        self.avg_frames = avg_frames
        
    def update(self):
        import time
        self.timestamps.append(time.time())
        
        # Keep only recent timestamps
        if len(self.timestamps) > self.avg_frames:
            self.timestamps.pop(0)
    
    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0
        
        # Calculate FPS based on time difference
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff > 0:
            return (len(self.timestamps) - 1) / time_diff
        return 0
    
    def draw(self, frame, position=(10, 30)):
        fps = self.get_fps()
        text = f"FPS: {fps:.1f}"
        
        # Choose color based on performance
        if fps >= 25:
            color = (0, 255, 0)  # Green
        elif fps >= 15:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        return frame
