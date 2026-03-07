"""
Main entry point for the football video enhancer.
Handles video capture, user input, and orchestrates processing.
"""

import cv2
import argparse
from config import *
from models.detector import PlayerDetector
from processing.effects import apply_translucency, create_debug_frame
from processing.tracker import PlayerTracker
from utils.visualization import draw_selection_overlay, FPSCounter, draw_instructions
import numpy as np


class AppState:
    """Global application state."""
    def __init__(self):
        self.effect_enabled = True          # translucency effect on/off
        self.selected_players = set()       # indices of manually selected players (detection indices)
        self.selected_tracked_ids = set()   # tracked IDs of manually selected players
        self.current_boxes = []              # bounding boxes of latest frame (for mouse interaction)
        self.current_masks = []               # segmentation masks (if any)
        self.frame_shape = None               # (height, width) of current frame
        self.tracker = None                   # Player tracker instance
        self.show_debug = False                # Toggle debug overlay
        self.fps_counter = FPSCounter()        # FPS counter


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks: select/deselect players using tracked IDs."""
    state = param  # AppState instance

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click falls inside any player's bounding box
        for i, box in enumerate(state.current_boxes):
            x1, y1, x2, y2 = map(int, box[:4])  # box format: [x1, y1, x2, y2]
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Get tracked ID if tracker exists
                if state.tracker and i in state.tracker.id_mapping:
                    tracked_id = state.tracker.id_mapping[i]
                    state.selected_tracked_ids.add(tracked_id)
                    # Also update detection indices for current frame
                    state.selected_players = state.tracker.get_detection_indices(state.selected_tracked_ids)
                    print(f"Player {tracked_id} (tracked) selected for translucency")
                else:
                    # Fallback to detection index
                    state.selected_players.add(i)
                    print(f"Player {i} (detection) selected for translucency")
                break
                
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right-click to deselect a specific player
        for i, box in enumerate(state.current_boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            if x1 <= x <= x2 and y1 <= y <= y2:
                if state.tracker and i in state.tracker.id_mapping:
                    tracked_id = state.tracker.id_mapping[i]
                    if tracked_id in state.selected_tracked_ids:
                        state.selected_tracked_ids.remove(tracked_id)
                        # Update detection indices
                        state.selected_players = state.tracker.get_detection_indices(state.selected_tracked_ids)
                        print(f"Player {tracked_id} deselected")
                else:
                    if i in state.selected_players:
                        state.selected_players.remove(i)
                        print(f"Player {i} deselected")
                break
                
    elif event == cv2.EVENT_MBUTTONDOWN:
        # Middle click to toggle debug view
        state.show_debug = not state.show_debug
        print(f"Debug view: {'ON' if state.show_debug else 'OFF'}")


def main():
    parser = argparse.ArgumentParser(description='Football video enhancer')
    parser.add_argument('--input', type=str, default=DEFAULT_VIDEO_PATH,
                        help='Path to input video file')
    parser.add_argument('--no-track', action='store_true',
                        help='Disable player tracking')
    parser.add_argument('--debug', action='store_true',
                        help='Start with debug overlay enabled')
    args = parser.parse_args()

    # Initialize video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.input}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fps:.2f} fps, {total_frames} frames")

    # Initialize detector
    detector = PlayerDetector(model_name=YOLO_MODEL_NAME, classes=DETECTION_CLASSES)
    
    # Initialize tracker (unless disabled)
    tracker = PlayerTracker() if not args.no_track else None

    # Application state
    state = AppState()
    state.tracker = tracker
    state.show_debug = args.debug

    # Setup OpenCV window and mouse callback
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)

    print("\n" + "="*50)
    print("CONTROLS:")
    print(f"  [{'T' if KEY_TOGGLE_EFFECT else 't'}] - Toggle effect ON/OFF")
    print(f"  Left-click on player - Select (make translucent)")
    print(f"  Right-click on player - Deselect")
    print(f"  Middle-click - Toggle debug overlay")
    print(f"  [{'D' if KEY_DESELECT_ALL else 'd'}] - Clear all selections")
    print(f"  [{'Q' if KEY_QUIT else 'q'}] - Quit")
    print("="*50 + "\n")

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video if it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        state.frame_shape = frame.shape[:2]  # (h, w)

        # Detect players and get masks
        boxes, masks = detector.detect(frame)
        
        # Update FPS counter
        state.fps_counter.update()

        # Update tracking if enabled
        if tracker and boxes is not None and len(boxes) > 0:
            # Update tracker with new detections
            id_mapping, tracked_boxes = tracker.update(boxes)
            
            # Update state with tracked boxes for mouse interaction
            state.current_boxes = tracked_boxes
            
            # Convert selected tracked IDs to current detection indices
            state.selected_players = tracker.get_detection_indices(state.selected_tracked_ids)
        else:
            state.current_boxes = boxes if boxes is not None else []
            if not tracker:
                # Without tracking, selected_players directly contains detection indices
                pass

        state.current_masks = masks

        # Process frame if effect is enabled
        if state.effect_enabled and boxes is not None and len(boxes) > 0:
            # Apply translucency based on distance and user selections
            output_frame = apply_translucency(
                frame, boxes, masks, state.selected_players, state.frame_shape
            )
        else:
            output_frame = frame.copy()

        # Draw visual indicators for selected players
        output_frame = draw_selection_overlay(
            output_frame, state.current_boxes, state.selected_players
        )

        # Add debug overlay if enabled
        if state.show_debug and boxes is not None and len(boxes) > 0:
            # Calculate opacities for debug display
            from processing.opacity import calculate_batch_opacity
            frame_area = state.frame_shape[0] * state.frame_shape[1]
            opacities = calculate_batch_opacity(boxes, frame_area)
            
            # Override for selected players
            for idx in state.selected_players:
                if idx < len(opacities):
                    opacities[idx] = SELECTED_PLAYER_OPACITY
            
            # Create debug frame
            output_frame = create_debug_frame(output_frame, boxes, masks, opacities, state.selected_players)
            
            # Add tracking info if available
            if tracker:
                y_offset = 30
                cv2.putText(output_frame, f"Tracked IDs: {len(tracker.tracked_players)}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(output_frame, f"Selected IDs: {state.selected_tracked_ids}", 
                           (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Add FPS counter
        output_frame = state.fps_counter.draw(output_frame, (10, 30))

        # Add instructions overlay
        output_frame = draw_instructions(
            output_frame, 
            state.effect_enabled, 
            len(state.selected_players)
        )

        # Add frame counter
        cv2.putText(output_frame, f"Frame: {frame_count}/{total_frames}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show the frame
        cv2.imshow(WINDOW_NAME, output_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_QUIT:
            break
        elif key == KEY_TOGGLE_EFFECT:
            state.effect_enabled = not state.effect_enabled
            print(f"Effect enabled: {state.effect_enabled}")
        elif key == KEY_DESELECT_ALL:
            state.selected_players.clear()
            state.selected_tracked_ids.clear()
            print("All manual selections cleared")
        elif key == ord('m'):  # 'm' for manual debug toggle
            state.show_debug = not state.show_debug
            print(f"Debug overlay: {'ON' if state.show_debug else 'OFF'}")
        elif key == ord('s'):  # 's' to save current frame
            cv2.imwrite(f"frame_{frame_count}.jpg", output_frame)
            print(f"Frame {frame_count} saved")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print(f"\nProcessed {frame_count} frames")
    if tracker:
        print(f"Tracked {len(tracker.tracked_players)} unique players throughout the video")


if __name__ == '__main__':
    main()