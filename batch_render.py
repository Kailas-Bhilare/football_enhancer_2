import cv2
import json
import argparse
from pathlib import Path
import numpy as np
import torch

from config import YOLO_MODEL_NAME, DETECTION_CLASSES
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

# ---------------------------------------------------------------------------
# Constants (UPDATED FOR DEBUG)
# ---------------------------------------------------------------------------

SD_BLEND_ALPHA = 0.6
SD_INTERVAL = 2
SD_MIN_MASK_PX = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_selection() -> set:
    path = Path("selection.json")
    if not path.exists():
        raise FileNotFoundError("selection.json not found.")
    with path.open("r") as f:
        return set(json.load(f)["selected_ids"])


def blend_sd_result(base, sd_output, mask, alpha=SD_BLEND_ALPHA):
    if sd_output is None:
        return base

    if sd_output.shape != base.shape:
        sd_output = cv2.resize(sd_output, (base.shape[1], base.shape[0]))

    feathered = feather_mask(mask, ksize=31)[..., None] * alpha

    result = (
        base.astype(np.float32) * (1.0 - feathered)
        + sd_output.astype(np.float32) * feathered
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="result.mp4")
    args = parser.parse_args()

    selected_ids = load_selection()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    detector = PlayerDetector(YOLO_MODEL_NAME, DETECTION_CLASSES)
    tracker = PlayerTracker()
    sd = SDInpainter()

    mask_smoother = TemporalMaskSmoother(history=5)
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
            reconstructor.update(
                frame,
                frame,
                np.zeros(frame.shape[:2], dtype=np.uint8)
            )
            writer.write(frame)
            continue

        tracker.update(boxes)
        selected_indices = tracker.get_detection_indices(selected_ids)

        mask = create_player_removal_mask(
            frame.shape[:2],
            boxes,
            None,
            selected_indices,
            auxiliary_masks=detector_masks,
        )

        mask = stabilize_mask(mask)
        mask = mask_smoother.smooth(mask)

        if np.sum(mask) > 0:

            # Clean mask
            mask_clean = cv2.medianBlur((mask * 255).astype(np.uint8), 5)
            mask255 = (mask_clean > 127).astype(np.uint8) * 255

            # Base inpaint
            base = cv2.inpaint(frame, mask255, 4, cv2.INPAINT_TELEA)

            # Reconstruction (still used as structure)
            reconstructed = reconstructor.reconstruct(frame, mask, base)

            if np.mean(reconstructed[mask > 0]) < 5:
                reconstructed = base

            # 🔥 FORCE SD (isolation test)
            run_sd = True

            if run_sd:
                sd_result = sd.inpaint(reconstructed, mask)

                if sd_result is not None:
                    output = blend_sd_result(reconstructed, sd_result, mask)
                    reconstructor.inject_clean_frame(output)
                else:
                    output = reconstructed

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                output = reconstructed

            output = composer.compose(frame, output, mask)
            reconstructor.update(frame, output, mask)

        else:
            output = frame.copy()
            reconstructor.update(frame, frame, mask)

        writer.write(output.astype(np.uint8))
        print(f"\rFrame {frame_idx}", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
