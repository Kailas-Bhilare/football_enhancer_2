# config.py

# --------------------------------------------------
# Detection
# --------------------------------------------------

# YOLO segmentation model
YOLO_MODEL_NAME = "yolov8m-seg.pt"

# Classes to detect (COCO: 0 = person)
DETECTION_CLASSES = [0]


# --------------------------------------------------
# General pipeline settings
# --------------------------------------------------

# Minimum pixels in mask to trigger SD
MIN_MASK_AREA = 50


# --------------------------------------------------
# Stable Diffusion settings
# --------------------------------------------------

SD_TARGET_SIZE = 512
SD_PAD_PX = 48

SD_PROMPT = (
    "empty football pitch grass, realistic stadium broadcast, "
    "clean field lines, natural lighting, high detail"
)

SD_NEGATIVE_PROMPT = (
    "players, people, athlete, blurry, distorted, warped lines, "
    "duplicate markings, artifacts, watermark"
)


# --------------------------------------------------
# Debug / toggles
# --------------------------------------------------

DEBUG = False