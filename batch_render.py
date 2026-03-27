import cv2
import json
import argparse
import numpy as np

from config import *
from models.detector import PlayerDetector
from processing.tracker import PlayerTracker
from processing.effects import create_player_removal_mask
from processing.sam_refiner import SAMRefiner
from processing.sd_inpainting import SDInpainter


def load_selection():
    with open("selection.json", "r") as f:
        return set(json.load(f)["selected_ids"])


# ---------------------------------------
# EDGE BLENDING (clean + stable)
# ---------------------------------------

def blend_edge(original, generated, mask):
    mask = (mask > 0).astype(np.uint8)

    # strong core removal
    core = cv2.erode(mask, np.ones((11, 11), np.uint8), 1)
    edge = mask - core

    edge = cv2.GaussianBlur(edge.astype(np.float32), (15, 15), 0)
    edge = edge[..., None]

    out = original.copy()

    # hard replace center
    out[core > 0] = generated[core > 0]

    # soft edges
    out = (
        generated.astype(np.float32) * edge +
        out.astype(np.float32) * (1 - edge)
    )

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------
# MAIN
# ---------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="result.mp4")
    args = parser.parse_args()

    selected_ids = load_selection()

    cap = cv2.VideoCapture(args.input)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    detector = PlayerDetector(YOLO_MODEL_NAME, DETECTION_CLASSES)
    tracker = PlayerTracker()
    sam = SAMRefiner()
    sd = SDInpainter()

    frame_idx = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        boxes, det_masks = detector.detect(frame)

        if boxes is None or len(boxes) == 0:
            writer.write(frame)
            continue

        tracker.update(boxes)
        selected_indices = tracker.get_detection_indices(selected_ids)

        if not selected_indices:
            writer.write(frame)
            continue

        # -------------------------
        # SAM MASK
        # -------------------------

        sam_masks = sam.refine(frame, boxes)
        if sam_masks is None:
            sam_masks = det_masks

        mask = create_player_removal_mask(
            frame.shape[:2],
            boxes,
            sam_masks,
            selected_indices,
            auxiliary_masks=det_masks,
        )

        # 🔥 expand mask (critical)
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), 1)

        if np.sum(mask) == 0:
            writer.write(frame)
            continue

        # -------------------------
        # SD PER FRAME (stable)
        # -------------------------

        sd_out = sd.inpaint(frame, mask)

        if sd_out is None:
            output = frame.copy()
        else:
            output = blend_edge(frame, sd_out, mask)

        writer.write(output.astype(np.uint8))

        print(f"\rFrame {frame_idx}", end="")

    cap.release()
    writer.release()

    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()