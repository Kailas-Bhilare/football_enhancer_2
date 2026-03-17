import cv2
import json
import argparse
import numpy as np
import torch

from config import *
from models.detector import PlayerDetector
from processing.tracker import PlayerTracker
from processing.effects import create_player_removal_mask
from processing.sd_inpainting import SDInpainter
from processing.sam_refiner import SAMRefiner


SD_INTERVAL = 12
MASK_HISTORY = 4
BLEND_ALPHA = 0.6


def load_selection():
    with open("selection.json", "r") as f:
        return set(json.load(f)["selected_ids"])


def smooth_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, 1)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
    return (mask > 0.25).astype(np.uint8)


def refine_mask(mask):
    kernel = np.ones((7, 7), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def temporal_mask(mask_history, new_mask):
    mask_history.append(new_mask)

    if len(mask_history) > MASK_HISTORY:
        mask_history.pop(0)

    combined = np.zeros_like(new_mask, dtype=np.float32)

    for i, m in enumerate(mask_history):
        weight = (i + 1) / len(mask_history)  # weighted smoothing
        combined += m * weight

    combined = combined / combined.max() if combined.max() > 0 else combined

    return (combined > 0.3).astype(np.uint8)


def blend_frames(prev, curr):
    if prev is None:
        return curr

    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]))

    return cv2.addWeighted(prev, BLEND_ALPHA, curr, 1 - BLEND_ALPHA, 0)


def normalize_frame(frame, width, height):
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame.astype(np.uint8)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="result.mp4")
    args = parser.parse_args()

    selected_ids = load_selection()
    print("Players selected:", selected_ids)

    cap = cv2.VideoCapture(args.input)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    detector = PlayerDetector(YOLO_MODEL_NAME, DETECTION_CLASSES)
    tracker = PlayerTracker()
    inpainter = SDInpainter()
    sam = SAMRefiner()

    mask_history = []
    prev_output = None

    frame_idx = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        boxes, _ = detector.detect(frame)

        if boxes is None or len(boxes) == 0:
            writer.write(frame)
            continue

        tracker.update(boxes)

        # SAM masks
        masks = sam.refine(frame, boxes)

        selected_indices = tracker.get_detection_indices(selected_ids)

        mask = create_player_removal_mask(
            frame.shape[:2],
            boxes,
            masks,
            selected_indices
        )

        # mask pipeline
        mask = smooth_mask(mask)
        mask = refine_mask(mask)
        mask = temporal_mask(mask_history, mask)

        if np.sum(mask) > 0:

            mask255 = (mask * 255).astype(np.uint8)

            # fast base inpainting
            output = cv2.inpaint(
                frame,
                mask255,
                3,
                cv2.INPAINT_TELEA
            )

            # SD refinement occasionally
            if frame_idx % SD_INTERVAL == 0:
                output = inpainter.inpaint(output, mask)
                torch.cuda.empty_cache()

        else:
            output = frame.copy()

        output = normalize_frame(output, width, height)

        # temporal frame blending (removes flicker trails)
        output = blend_frames(prev_output, output)
        prev_output = output.copy()

        writer.write(output)

        percent = frame_idx / total * 100
        print(f"\rProcessing {percent:5.1f}%  Frame {frame_idx}/{total}", end="")

    cap.release()
    writer.release()

    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()