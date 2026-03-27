import cv2
import json
import argparse
import numpy as np
import torch

from config import YOLO_MODEL_NAME, DETECTION_CLASSES
from models.detector import PlayerDetector
from processing.tracker import PlayerTracker
from processing.sam_refiner import SAMRefiner
from processing.sd_inpainting import SDInpainter


def load_selection():
    with open("selection.json", "r") as f:
        return set(json.load(f)["selected_ids"])


def blend_edge(original, generated, mask):
    mask = (mask > 0).astype(np.uint8)

    if generated.shape[:2] != original.shape[:2]:
        generated = cv2.resize(generated, (original.shape[1], original.shape[0]))

    core = cv2.erode(mask, np.ones((11, 11), np.uint8), 1)
    edge = (mask - core).astype(np.uint8)

    edge_alpha = cv2.GaussianBlur(edge.astype(np.float32), (17, 17), 0)
    edge_alpha = np.clip(edge_alpha, 0.0, 1.0)[..., None]

    out = original.copy()
    out[core > 0] = generated[core > 0]

    out = (
        generated.astype(np.float32) * edge_alpha
        + out.astype(np.float32) * (1.0 - edge_alpha)
    )

    return np.clip(out, 0, 255).astype(np.uint8)


def temporal_stabilize(prev, curr, mask, alpha=0.12):
    if prev is None:
        return curr

    mask = (mask > 0).astype(np.uint8)
    core = cv2.erode(mask, np.ones((9, 9), np.uint8), 1)

    if np.count_nonzero(core) == 0:
        return curr

    alpha_map = core[..., None].astype(np.float32) * alpha
    out = curr.astype(np.float32) * (1.0 - alpha_map) + prev.astype(np.float32) * alpha_map
    return np.clip(out, 0, 255).astype(np.uint8)


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
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {args.output}")

    detector = PlayerDetector(YOLO_MODEL_NAME, DETECTION_CLASSES)
    tracker = PlayerTracker()
    sam = SAMRefiner()
    sd = SDInpainter()

    prev_output = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        boxes, det_masks = detector.detect(frame)

        if boxes is None or len(boxes) == 0:
            tracker.update([])
            prev_output = frame.copy()
            writer.write(frame)
            print(f"\rFrame {frame_idx}", end="", flush=True)
            continue

        tracker.update(boxes)
        selected_indices = tracker.get_detection_indices(selected_ids)

        if not selected_indices:
            prev_output = frame.copy()
            writer.write(frame)
            print(f"\rFrame {frame_idx}", end="", flush=True)
            continue

        sam_masks = sam.refine(frame, boxes)
        if sam_masks is None:
            sam_masks = det_masks

        output = frame.copy()

        # PROCESS EACH PLAYER SEPARATELY
        for idx in selected_indices:

            if idx >= len(sam_masks):
                continue

            player_mask = sam_masks[idx]

            if player_mask.shape != frame.shape[:2]:
                player_mask = cv2.resize(player_mask, (frame.shape[1], frame.shape[0]))

            # small dilation per player ONLY
            player_mask = cv2.dilate(player_mask, np.ones((5, 5), np.uint8), 1)

            if np.count_nonzero(player_mask) == 0:
                continue

            sd_out = sd.inpaint(output, player_mask)

            if sd_out is not None:
                output = blend_edge(output, sd_out, player_mask)

        # LIGHT TEMPORAL STABILIZATION (combined mask)
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for idx in selected_indices:
            if idx < len(sam_masks):
                combined_mask = np.maximum(combined_mask, sam_masks[idx])

        output = temporal_stabilize(prev_output, output, combined_mask)

        prev_output = output.copy()
        writer.write(output.astype(np.uint8))

        print(f"\rFrame {frame_idx}", end="", flush=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cap.release()
    writer.release()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()