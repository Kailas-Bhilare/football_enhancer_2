import cv2
import json
import argparse
import numpy as np

from config import YOLO_MODEL_NAME, DETECTION_CLASSES
from models.detector import PlayerDetector
from processing.tracker import PlayerTracker
from processing.sam_refiner import SAMRefiner
from processing.sd_inpainting import SDInpainter


def load_selection():
    with open("selection.json", "r") as f:
        return set(json.load(f)["selected_ids"])


# -------------------------
# HOMOGRAPHY (stable camera motion)
# -------------------------
def warp_previous(prev, curr):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    pts_prev = cv2.goodFeaturesToTrack(prev_gray, 2000, 0.01, 7)

    if pts_prev is None:
        return prev

    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts_prev, None
    )

    pts_prev = pts_prev[status == 1]
    pts_curr = pts_curr[status == 1]

    if len(pts_prev) < 10:
        return prev

    H, _ = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, 5.0)

    if H is None:
        return prev

    h, w = prev.shape[:2]
    return cv2.warpPerspective(prev, H, (w, h))


# -------------------------
# MASK REFINEMENT (fix gaps)
# -------------------------
def refine_mask(mask):
    mask = (mask > 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    # fill gaps between limbs
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # slight expansion
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), 1)

    return mask


# -------------------------
# CLEAN EDGE BLENDING
# -------------------------
def blend_edge(original, generated, mask):

    mask = (mask > 0).astype(np.uint8)

    core = cv2.erode(mask, np.ones((13, 13), np.uint8), 1)
    edge = mask - core

    # distance-based blending (cleaner than blur)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    alpha = np.clip(dist / 5.0, 0, 1)

    alpha = alpha * edge
    alpha = alpha[..., None]

    out = original.copy()

    # hard replace center
    out[core > 0] = generated[core > 0]

    # blend only edges
    out = (
        generated.astype(np.float32) * alpha +
        out.astype(np.float32) * (1 - alpha)
    )

    return np.clip(out, 0, 255).astype(np.uint8)


def player_area(box):
    x1, y1, x2, y2 = map(float, box[:4])
    return max(1.0, (x2 - x1) * (y2 - y1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="result.mp4")
    args = parser.parse_args()

    selected_ids = load_selection()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.input}")

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5) or 30.0

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

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
            continue

        tracker.update(boxes)
        selected_indices = tracker.get_detection_indices(selected_ids)

        if not selected_indices:
            prev_output = frame.copy()
            writer.write(frame)
            continue

        sam_masks = sam.refine(frame, boxes)
        if sam_masks is None:
            sam_masks = det_masks

        output = frame.copy()

        # process big players first
        selected_indices = sorted(
            selected_indices,
            key=lambda i: player_area(boxes[i]),
            reverse=True,
        )

        for idx in selected_indices:

            if idx >= len(sam_masks):
                continue

            player_mask = sam_masks[idx]

            if player_mask.shape != frame.shape[:2]:
                player_mask = cv2.resize(
                    player_mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # improved mask
            player_mask = refine_mask(player_mask)

            if np.count_nonzero(player_mask) == 0:
                continue

            if prev_output is not None:

                warped = warp_previous(prev_output, frame)

                fill = warped.copy()
                fill[player_mask > 0] = warped[player_mask > 0]

                # less SD usage
                if np.mean(fill[player_mask > 0]) < 25:

                    sd_out = sd.inpaint(output, player_mask)

                    if sd_out is not None:
                        output = blend_edge(output, sd_out, player_mask)
                    else:
                        mask255 = (player_mask * 255).astype(np.uint8)
                        output = cv2.inpaint(output, mask255, 3, cv2.INPAINT_TELEA)

                else:
                    output[player_mask > 0] = fill[player_mask > 0]

            else:
                sd_out = sd.inpaint(output, player_mask)
                if sd_out is not None:
                    output = blend_edge(output, sd_out, player_mask)

        # subtle realism polish
        output = cv2.bilateralFilter(output, 5, 20, 20)

        noise = np.random.normal(0, 2, output.shape).astype(np.int16)
        output = np.clip(output.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        prev_output = output.copy()
        writer.write(output)

        print(f"\rFrame {frame_idx}", end="")

    cap.release()
    writer.release()
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()