import cv2
import os
import json
import argparse
import numpy as np

from config import *
from models.detector import PlayerDetector
from processing.tracker import PlayerTracker
from processing.effects import create_player_removal_mask
from processing.sd_inpainting import SDInpainter


def load_selection():
    with open("selection.json", "r") as f:
        return set(json.load(f)["selected_ids"])


def smooth_mask(mask):
    mask = cv2.dilate(mask, np.ones((5,5),np.uint8),1)
    mask = cv2.GaussianBlur(mask,(15,15),0)
    return (mask>0.25).astype(np.uint8)


def resize_for_sd(frame, mask, size=256):

    h, w = frame.shape[:2]

    frame_small = cv2.resize(frame, (size, size))
    mask_small = cv2.resize(mask, (size, size))

    return frame_small, mask_small, (w, h)


def upscale_back(frame_small, orig_size):
    return cv2.resize(frame_small, orig_size)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="result.mp4")

    args = parser.parse_args()

    selected_ids = load_selection()

    cap = cv2.VideoCapture(args.input)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    detector = PlayerDetector(YOLO_MODEL_NAME, DETECTION_CLASSES)
    tracker = PlayerTracker()
    sd = SDInpainter()

    prev_output = None
    frame_idx = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        boxes, masks = detector.detect(frame)

        if boxes is None or len(boxes) == 0:
            writer.write(frame)
            continue

        tracker.update(boxes)

        selected_indices = tracker.get_detection_indices(selected_ids)

        mask = create_player_removal_mask(
            frame.shape[:2],
            boxes,
            masks,
            selected_indices
        )

        mask = smooth_mask(mask)

        if np.sum(mask) > 0:

            small_frame, small_mask, orig_size = resize_for_sd(frame, mask)

            output_small = sd.inpaint(small_frame, small_mask)

            output = upscale_back(output_small, orig_size)

        else:
            output = frame.copy()

        # temporal smoothing
        if prev_output is not None:
            output = cv2.addWeighted(prev_output, 0.7, output, 0.3, 0)

        prev_output = output.copy()

        writer.write(output.astype(np.uint8))

        print(f"\rFrame {frame_idx}", end="")

    cap.release()
    writer.release()

    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
