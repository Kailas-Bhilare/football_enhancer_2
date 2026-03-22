"""
render_video.py

Two-step rendering pipeline:
  1. Player selection  (main.py)
  2. Background reconstruction + inpainting  (this file)

Reconstruction pipeline per frame
----------------------------------
  a. Detect players (YOLO) + refine masks (SAM when available).
  b. Build removal mask → stabilise → temporally smooth.
  c. Classical inpaint (TELEA) as fast fallback.
  d. MotionCompensatedBackgroundReconstructor fills mask from aligned buffer.
  e. SD inpainting runs every SD_INTERVAL frames on large masks; result is
     injected back into the reconstructor buffer so future frames receive it
     through the same homography-alignment path — no separate stale cache.
  f. TemporalRemovalComposer smooths per-frame flicker.
  g. Raw frame (not output) is stored in the reconstructor buffer to prevent
     artifact feedback loops.
"""

import cv2
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch

from config import (
    YOLO_MODEL_NAME,
    DETECTION_CLASSES,
)
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASK_HISTORY = 6

# Run SD every N frames when the mask is large enough.
# SD output is injected into the background buffer, so the result persists
# across future frames via homography alignment — no stale cache needed.
SD_INTERVAL = 8
SD_MIN_MASK_PIXELS = 400

SD_BLEND_ALPHA = 0.55  # How strongly to blend SD result over reconstruction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_selection() -> set:
    path = Path("selection.json")
    if not path.exists():
        raise FileNotFoundError(
            "selection.json not found. Run the selection step first."
        )
    with path.open("r") as f:
        return set(json.load(f)["selected_ids"])


def normalize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))
    return frame.astype(np.uint8)


def prepare_debug_directory(debug_dir, clean: bool = False) -> Path:
    p = Path(debug_dir)
    if clean and p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_debug_frame(debug_dir: Path, frame_idx: int, frame: np.ndarray) -> Path:
    fp = Path(debug_dir) / f"frame_{frame_idx:04d}.png"
    if not cv2.imwrite(str(fp), frame):
        raise RuntimeError(f"Failed to write debug frame: {fp}")
    return fp


def blend_sd_into_reconstruction(
    reconstruction: np.ndarray,
    sd_output: np.ndarray,
    mask: np.ndarray,
    alpha: float = SD_BLEND_ALPHA,
) -> np.ndarray:
    """
    Blend Stable Diffusion output over the background reconstruction inside
    the mask region with a feathered edge.
    """
    if sd_output is None:
        return reconstruction
    if sd_output.shape != reconstruction.shape:
        sd_output = cv2.resize(
            sd_output, (reconstruction.shape[1], reconstruction.shape[0])
        )
    feathered = feather_mask(mask, ksize=25)[..., None] * alpha
    blended = (
        reconstruction.astype(np.float32) * (1.0 - feathered)
        + sd_output.astype(np.float32) * feathered
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove selected players and reconstruct background."
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="result.mp4", help="Output video path")
    parser.add_argument("--debug", action="store_true", help="Export frame images instead of video")
    parser.add_argument("--debug-dir", default="debug", help="Directory for debug frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
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
        print(f"Debug mode: saving up to {args.max_frames or '∞'} frames → {debug_path}/")
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

    # ── Models ──────────────────────────────────────────────────────────────
    detector = PlayerDetector(YOLO_MODEL_NAME, DETECTION_CLASSES)
    tracker  = PlayerTracker()
    sam      = SAMRefiner()
    sd       = SDInpainter()

    # ── Processing components ────────────────────────────────────────────────
    mask_smoother = TemporalMaskSmoother(history=MASK_HISTORY)
    composer      = TemporalRemovalComposer(blend_alpha=0.15, feather_radius=21)
    reconstructor = MotionCompensatedBackgroundReconstructor()

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        boxes, detector_masks = detector.detect(frame)

        # ── No players detected ──────────────────────────────────────────────
        if boxes is None or len(boxes) == 0:
            # Store raw frame with empty mask so the buffer accumulates clean
            # background even during player-free intervals.
            reconstructor.update(frame, frame, np.zeros(frame.shape[:2], dtype=np.uint8))
            output = frame.copy()

        # ── Players detected ─────────────────────────────────────────────────
        else:
            tracker.update(boxes)

            sam_masks        = sam.refine(frame, boxes)
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
                # ── Step 1: Classical inpaint ──────────────────────────────
                mask_clean = cv2.medianBlur((mask * 255).astype(np.uint8), 5)
                mask255    = (mask_clean > 127).astype(np.uint8) * 255
                base       = cv2.inpaint(frame, mask255, 4, cv2.INPAINT_TELEA)

                # ── Step 2: Motion-compensated background reconstruction ───
                # reconstruct() fills mask pixels from the homography-aligned
                # frame buffer, falling back to colour-corrected TELEA base.
                reconstructed = reconstructor.reconstruct(frame, mask, base)

                # Safety check: implausibly dark output means all-black border
                # padding leaked through — revert to TELEA base for those pixels.
                if np.mean(reconstructed[mask > 0]) < 10:
                    reconstructed = base

                # ── Step 3: Stable Diffusion refinement ───────────────────
                # Run SD on a cadence; inject the result back into the
                # reconstructor buffer (mask=zeros) so future frames reuse it
                # through homography alignment — no stale blending needed.
                run_sd = (
                    sd.enabled
                    and (frame_idx % SD_INTERVAL == 0)
                    and (int(np.sum(mask)) > SD_MIN_MASK_PIXELS)
                )

                if run_sd:
                    sd_result = sd.inpaint(reconstructed, mask)
                    if sd_result is not None:
                        # Blend SD result into the reconstruction for this frame
                        output = blend_sd_into_reconstruction(
                            reconstructed, sd_result, mask
                        )
                        # Inject the blended result as a clean background frame.
                        # mask=zeros tells the buffer: every pixel here is valid
                        # background (no players) and may be used by future frames.
                        reconstructor.inject_clean_frame(output)
                    else:
                        output = reconstructed

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    output = reconstructed

                # ── Step 4: Temporal smoothing ─────────────────────────────
                output = composer.compose(frame, output, mask)

                # ── Step 5: Update background buffer ──────────────────────
                # Store the RAW frame + its occlusion mask.
                # The processed `output` is NOT stored — it may contain
                # residual inpainting artefacts that would corrupt future
                # reconstructions if fed back as background.
                reconstructor.update(frame, output, mask)

            else:
                output = frame.copy()
                # No players this frame → fully clean; store as background.
                reconstructor.update(frame, frame, mask)

        output = normalize_frame(output, width, height)

        if args.debug:
            save_debug_frame(debug_path, frame_idx, output)
        else:
            writer.write(output)

        pct = frame_idx / total * 100
        print(f"\rProcessing {pct:5.1f}%  frame {frame_idx}/{total}", end="", flush=True)

        if args.debug and args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    if writer:
        writer.release()

    if args.debug:
        print(f"\nDebug frames saved → {debug_path}")
    else:
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
