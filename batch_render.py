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
# 🔥 NEW: optical flow warp
# -------------------------
def warp_previous(prev, curr):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )

    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    warped = cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR)

    return warped


def blend_edge(original, generated, mask):
    mask = (mask > 0).astype(np.uint8)

    core = cv2.erode(mask, np.ones((11, 11), np.uint8), 1)
    edge = mask - core

    edge = cv2.GaussianBlur(edge.astype(np.float32), (17, 17), 0)[..., None]

    out = original.copy()
    out[core > 0] = generated[core > 0]

    out = (
        generated.astype(np.float32) * edge +
        out.astype(np.float32) * (1 - edge)
    ).astype(np.uint8)

    return out


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

            player_mask = cv2.dilate(player_mask, np.ones((5, 5), np.uint8), 1)

            if np.count_nonzero(player_mask) == 0:
                continue

            # ---------------------------------
            # 🔥 KEY FIX: reuse previous frame
            # ---------------------------------
            if prev_output is not None:

                warped = warp_previous(prev_output, frame)

                fill = warped.copy()

                # use real pixels instead of SD
                fill[player_mask > 0] = warped[player_mask > 0]

                # detect if warped failed (black / invalid)
                if np.mean(fill[player_mask > 0]) < 10:

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

        prev_output = output.copy()
        writer.write(output.astype(np.uint8))

        print(f"\rFrame {frame_idx}", end="")

    cap.release()
    writer.release()
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()