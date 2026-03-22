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

# Run SD inpainting every N frames when the mask is large enough.
# Lowered from 15 → 8 so the generative result stays fresh without
# letting OpenCV-only frames dominate the background model.
SD_INTERVAL = 8
SD_MIN_MASK_PIXELS = 500


def load_selection():
    selection_path = Path("selection.json")
    if not selection_path.exists():
        raise FileNotFoundError(
            "selection.json not found. Run the selection step first."
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
        0, 255,
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

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (args.max_frames or 1)

    writer = None
    debug_path = None

    if args.debug:
        debug_path = prepare_debug_directory(args.debug_dir)
        print(f"Debug mode: saving up to {args.max_frames or 10} frames → {debug_path}/")
    else:
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {args.output}")

    detector = PlayerDetector(YOLO_MODEL_NAME, DETECTION_CLASSES)
    tracker  = PlayerTracker()
    sam      = SAMRefiner()
    sd       = SDInpainter()

    mask_smoother = TemporalMaskSmoother(history=MASK_HISTORY)
    composer      = TemporalRemovalComposer(blend_alpha=0.15, feather_radius=21)
    reconstructor = MotionCompensatedBackgroundReconstructor()

    # Cache the most-recent SD result so we can blend it on non-SD frames
    # instead of dropping back to raw OpenCV inpainting artifacts.
    last_sd_output = None
    last_sd_mask   = None

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        boxes, detector_masks = detector.detect(frame)

        # ── No detections ──────────────────────────────────────────────
        if boxes is None or len(boxes) == 0:
            # BUG FIX: update reconstructor with the RAW frame (not output)
            # so genuine background pixels enter the buffer.
            reconstructor.update(frame, frame, np.zeros(frame.shape[:2], dtype=np.uint8))
            output = frame.copy()

        # ── Players detected ───────────────────────────────────────────
        else:
            tracker.update(boxes)
            sam_masks        = sam.refine(frame, boxes)
            selected_indices = tracker.get_detection_indices(selected_ids)

            mask = create_player_removal_mask(
                frame.shape[:2], boxes, sam_masks,
                selected_indices, auxiliary_masks=detector_masks,
            )
            mask = stabilize_mask(mask)
            mask = mask_smoother.smooth(mask)

            if np.sum(mask) > 0:
                # ── Step 1: classical inpaint (fast, always-available base) ──
                mask_clean = cv2.medianBlur((mask * 255).astype(np.uint8), 5)
                mask255    = (mask_clean > 127).astype(np.uint8) * 255
                base       = cv2.inpaint(frame, mask255, 4, cv2.INPAINT_TELEA)

                # ── Step 2: background reconstruction from frame buffer ──
                # BUG FIX: pass `base` (not `frame`) so the reconstructor has
                # a valid per-pixel fallback for every position, but source
                # of background data is the clean rolling buffer — not the
                # artefact-laden output of a previous iteration.
                reconstructed = reconstructor.reconstruct(frame, mask, base)

                # Sanity check: if reconstructed mask region is implausibly
                # dark (all-black warp padding crept in), fall back to base.
                if np.mean(reconstructed[mask > 0]) < 10:
                    reconstructed = base

                # ── Step 3: Stable Diffusion refinement ──────────────────
                # Run SD on a fixed interval; between SD frames reuse the
                # cached result so we never display raw OpenCV artefacts.
                # BUG FIX (original): SD ran every 15 frames but the 14
                # in-between frames used only OpenCV inpainting, which was
                # then baked into the background model as "good" background.
                run_sd = (
                    sd.enabled
                    and frame_idx % SD_INTERVAL == 0
                    and np.sum(mask) > SD_MIN_MASK_PIXELS
                )

                if run_sd:
                    sd_result = sd.inpaint(reconstructed, mask)
                    if sd_result is not None:
                        last_sd_output = sd_result
                        last_sd_mask   = mask.copy()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Blend SD result (cached or fresh) with the reconstruction
                if last_sd_output is not None and last_sd_mask is not None:
                    # Warp the cached SD result into the current mask region
                    # using a simple alpha blend weighted by mask overlap
                    overlap = np.minimum(mask.astype(np.float32),
                                         last_sd_mask.astype(np.float32))
                    if overlap.sum() > 0:
                        output = blend_sd_result(reconstructed, last_sd_output,
                                                 overlap, alpha=SD_BLEND_ALPHA)
                    else:
                        output = reconstructed
                else:
                    output = reconstructed

                # ── Step 4: temporal smoothing of the final composite ──
                output = composer.compose(frame, output, mask)

                # ── Step 5: update background buffer ─────────────────────
                # BUG FIX: store the RAW frame + its occlusion mask, NOT the
                # processed output.  Storing `output` fed inpainting artefacts
                # back into future reconstructions.
                reconstructor.update(frame, output, mask)

            else:
                output = frame.copy()
                reconstructor.update(frame, frame, mask)

        output = normalize_frame(output, width, height)

        if args.debug:
            save_debug_frame(debug_path, frame_idx, output)
        else:
            writer.write(output)

        percent = frame_idx / total * 100
        print(f"\rProcessing {percent:5.1f}%  frame {frame_idx}/{total}", end="")

        if args.debug and args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    if writer:
        writer.release()

    if args.debug:
        print(f"\nDebug frames saved → {debug_path}")
    else:
        print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
