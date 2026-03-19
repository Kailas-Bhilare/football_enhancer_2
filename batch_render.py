import cv2
import json
import argparse
import numpy as np

from config import *
from models.detector import PlayerDetector
from processing.tracker import PlayerTracker
from processing.effects import (
    MotionCompensatedBackgroundReconstructor,
    TemporalMaskSmoother,
    TemporalRemovalComposer,
    create_player_removal_mask,
    stabilize_mask,
)
from processing.sd_inpainting import SDInpainter


SD_BLEND_ALPHA = 0.2


def load_selection():
    with open("selection.json", "r") as f:
        return set(json.load(f)["selected_ids"])


def resize_for_sd(frame, mask, size=256):

    h, w = frame.shape[:2]

    frame_small = cv2.resize(frame, (size, size))
    mask_small = cv2.resize(mask, (size, size))

    return frame_small, mask_small, (w, h)


def upscale_back(frame_small, orig_size):
    return cv2.resize(frame_small, orig_size)


def blend_sd_result(base_output, sd_output, mask, alpha=SD_BLEND_ALPHA):
    mask = (mask > 0).astype(np.float32)[..., None]
    refined = cv2.addWeighted(base_output, 1.0 - alpha, sd_output, alpha, 0)
    output = base_output.astype(np.float32) * (1.0 - mask) + refined.astype(np.float32) * mask
    return np.clip(output, 0, 255).astype(np.uint8)


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

    # Updated flicker-reduction settings
    mask_smoother = TemporalMaskSmoother(history=5)
    composer = TemporalRemovalComposer(blend_alpha=0.35, feather_radius=31)
    reconstructor = MotionCompensatedBackgroundReconstructor()

    frame_idx = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        boxes, detector_masks = detector.detect(frame)

        if boxes is None or len(boxes) == 0:
            output = frame.copy()
            reconstructor.update(frame, output, np.zeros(frame.shape[:2], dtype=np.uint8))
            writer.write(output)
            continue

        tracker.update(boxes)

        selected_indices = tracker.get_detection_indices(selected_ids)

        mask = create_player_removal_mask(
            frame.shape[:2],
            boxes,
            detector_masks,
            selected_indices,
        )

        mask = stabilize_mask(mask)
        mask = mask_smoother.smooth(mask)

        if np.sum(mask) > 0:

            small_frame, small_mask, orig_size = resize_for_sd(frame, mask)
            output_small = sd.inpaint(small_frame, small_mask)
            output = upscale_back(output_small, orig_size)
            output = blend_sd_result(frame, output, mask)
            output = reconstructor.reconstruct(frame, mask, output)
            output = composer.compose(frame, output, mask)
            reconstructor.update(frame, output, mask)

        else:
            output = frame.copy()
            reconstructor.update(frame, output, mask)

        writer.write(output.astype(np.uint8))

        print(f"\rFrame {frame_idx}", end="")

    cap.release()
    writer.release()

    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()