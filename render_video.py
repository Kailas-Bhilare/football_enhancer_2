import cv2
import json
import argparse
import numpy as np
import torch

from config import *
from models.detector import PlayerDetector
from processing.tracker import PlayerTracker
from processing.effects import (
    MotionCompensatedBackgroundReconstructor,
    TemporalMaskSmoother,
    TemporalRemovalComposer,
    create_player_removal_mask,
    stabilize_mask,
    feather_mask,
)
from processing.sd_inpainting import SDInpainter
from processing.sam_refiner import SAMRefiner


MASK_HISTORY = 5
SD_BLEND_ALPHA = 0.85


def load_selection():
    with open("selection.json", "r") as f:
        return set(json.load(f)["selected_ids"])


def normalize_frame(frame, width, height):
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame.astype(np.uint8)


def blend_sd_result(base_output, sd_output, mask, alpha=SD_BLEND_ALPHA):
    if sd_output is None:
        return base_output

    if sd_output.shape != base_output.shape:
        sd_output = cv2.resize(sd_output, (base_output.shape[1], base_output.shape[0]))

    mask_alpha = feather_mask(mask, 31)[..., None] * alpha
    output = base_output.astype(np.float32) * (1.0 - mask_alpha)
    output += sd_output.astype(np.float32) * mask_alpha
    return np.clip(output, 0, 255).astype(np.uint8)


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

    mask_smoother = TemporalMaskSmoother(history=MASK_HISTORY)
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

        sam_masks = sam.refine(frame, boxes)
        selected_indices = tracker.get_detection_indices(selected_ids)

        mask = create_player_removal_mask(
            frame.shape[:2],
            boxes,
            sam_masks,
            selected_indices,
            auxiliary_masks=detector_masks,
        )

        mask = stabilize_mask(mask)
        mask = mask_smoother.smooth(mask)

        if np.sum(mask) > 0:

            mask255 = (mask * 255).astype(np.uint8)

            base_output = cv2.inpaint(
                frame,
                mask255,
                5,
                cv2.INPAINT_TELEA
            )

            reconstructed = reconstructor.reconstruct(frame, mask, base_output)
            sd_output = inpainter.inpaint(reconstructed, mask)
            output = blend_sd_result(reconstructed, sd_output, mask)
            torch.cuda.empty_cache()

            output = reconstructor.reconstruct(frame, mask, output)
            output = composer.compose(frame, output, mask)
            reconstructor.update(frame, output, mask)

        else:
            output = frame.copy()
            reconstructor.update(frame, output, mask)

        output = normalize_frame(output, width, height)
        writer.write(output)

        percent = frame_idx / total * 100 if total else 0
        print(f"\rProcessing {percent:5.1f}%  Frame {frame_idx}/{total}", end="")

    cap.release()
    writer.release()

    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()