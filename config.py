"""
Configuration parameters for the football video enhancer.
All values can be tweaked to match your preferences.
"""

# Model settings
YOLO_MODEL_NAME = 'yolov8m-seg.pt'   # or 'yolov8n-seg.pt' for faster speed
DETECTION_CLASSES = [0]              # COCO class 0 = person

# Opacity mapping (based on bounding box area relative to frame)
OPACITY_MIN = 0.8                     # Most distant players (higher = more visible)
OPACITY_MAX = 1.0                     # Closest players (fully visible)
AREA_RATIO_FOR_MAX_OPACITY = 0.15     # When bbox covers 15% of frame → fully opaque
AREA_RATIO_FOR_MIN_OPACITY = 0.01     # When bbox covers 1% of frame → min opacity

# Manual selection opacity (when user clicks a player)
SELECTED_PLAYER_OPACITY = 0.2          # Make selected players very translucent (0.2 = 20% visible)

# Inpainting settings
INPAINTING_METHOD = 'telea'            # 'telea' or 'ns'
INPAINTING_RADIUS = 3                  # Radius for inpainting

# Key bindings
KEY_QUIT = ord('q')
KEY_TOGGLE_EFFECT = ord('t')
KEY_DESELECT_ALL = ord('d')            # Clear all selected players
KEY_DEBUG_TOGGLE = ord('m')            # Toggle debug overlay
KEY_SAVE_FRAME = ord('s')               # Save current frame

# Video source (can be overridden by command line)
DEFAULT_VIDEO_PATH = 'data/sample.mp4'

# Display window name
WINDOW_NAME = 'Football View Enhancer'