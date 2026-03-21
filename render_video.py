import cv2
import json
import argparse
import shutil
from pathlib import Path
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


MASK_HISTORY = 6
SD_BLEND_ALPHA = 0.45


def load_selection():
    selection_path = Path("selection.json")
    if not selection_path.exists():
        raise FileNotFoundError(
            "selection.json not found. Run the selection step first or provide the file."
        )

    with selection_path.open("r") as f:
        return set(json.load(f)["selected_ids"])


def normalize_frame(frame, width, height):
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))
    return frame.astype(np.uint8)


def prepare_debug_directory(debug_dir, clean=False):
    debug_path = Path(debug_dir)

    if clean and debug_path.exists():
        shutil.rmtree(debug_path)

    debug_path.mkdir(parents=True, exist_ok=True)
    return debug_path


def save_debug_frame(debug_dir, frame_idx, frame):
    frame_path = Path(debug_dir) / f"frame_{frame_idx:04d}.png"
    success = cv2.imwrite(str(frame_path), frame)

    if not success:
        raise RuntimeError(f"Failed to write debug frame: {frame_path}")

    return frame_path


def blend_sd_result(base, sd, mask, alpha=SD_BLEND_ALPHA):

    if sd is None:
        return base

    if sd.shape != base.shape:
        sd = cv2.resize(sd, (base.shape[1], base.shape[0]))

    mask_alpha = feather_mask(mask, 25)[..., None] * alpha

    return np.clip(
        base.astype(np.float32) * (1 - mask_alpha) +
        sd.astype(np.float32) * mask_alpha,
        0, 255
    ).astype(np.uint8)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="result.mp4")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", default="debug")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    selected_ids = load_selection()
    print("Players selected:", selected_ids)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0
    if total <= 0:
        total = args.max_frames or 1

    writer = None
    debug_path = None

    if args.debug:
        debug_path = prepare_debug_directory(args.debug_dir)
        print(f"Debug mode: exporting up to {args.max_frames or 10} frames to {debug_path}/")
    else:
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {args.output}")

    detector = PlayerDetector(YOLO_MODEL_NAME, DETECTION_CLASSES)
    tracker = PlayerTracker()
    sam = SAMRefiner()
    sd = SDInpainter()

    mask_smoother = TemporalMaskSmoother(history=MASK_HISTORY)
    composer = TemporalRemovalComposer(blend_alpha=0.15, feather_radius=21)
    reconstructor = MotionCompensatedBackgroundReconstructor()

    frame_idx = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        boxes, detector_masks = detector.detect(frame)

        if boxes is None or len(boxes) == 0:
            reconstructor.update(frame, frame, np.zeros(frame.shape[:2], dtype=np.uint8))
            output = frame.copy()
        else:
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

                mask_clean = cv2.medianBlur((mask * 255).astype(np.uint8), 5)
                mask255 = (mask_clean > 127).astype(np.uint8) * 255

                base = cv2.inpaint(frame, mask255, 4, cv2.INPAINT_TELEA)
                reconstructed = reconstructor.reconstruct(frame, mask, base)

                if np.mean(reconstructed[mask > 0]) < 10:
                    reconstructed = base

                if frame_idx % 15 == 0 and np.sum(mask) > 500:
                    sd_output = sd.inpaint(reconstructed, mask)
                    output = blend_sd_result(reconstructed, sd_output, mask)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    output = reconstructed

                output = composer.compose(frame, output, mask)
                reconstructor.update(frame, output, mask)

            else:
                output = frame.copy()
                reconstructor.update(frame, output, mask)

        output = normalize_frame(output, width, height)

        if args.debug:
            save_debug_frame(debug_path, frame_idx, output)
        else:
            writer.write(output)

        percent = frame_idx / total * 100
        print(f"\rProcessing {percent:5.1f}% Frame {frame_idx}/{total}", end="")

        if args.debug and args.max_frames is not None and frame_idx >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    if args.debug:
        print(f"\nSaved debug frames to: {debug_path}")
    else:
        print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
