"""
Mask utilities for player removal.

Goal:
- keep masks tight
- avoid box-shaped artifacts
- preserve thin field lines as much as possible
"""

from collections import deque

import cv2
import numpy as np


def feather_mask(mask: np.ndarray, ksize: int = 21) -> np.ndarray:
    """
    Soft alpha from a binary mask.
    """
    mask = (mask > 0).astype(np.float32)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _span_fill(mask: np.ndarray) -> np.ndarray:
    """
    Fill gaps across rows and columns inside a fragmented silhouette.
    """
    filled = (mask > 0).astype(np.uint8)
    if filled.ndim != 2 or np.count_nonzero(filled) == 0:
        return filled

    row_hits = np.where(filled.any(axis=1))[0]
    for row in row_hits:
        cols = np.where(filled[row] > 0)[0]
        if cols.size >= 2:
            filled[row, cols[0] : cols[-1] + 1] = 1

    col_hits = np.where(filled.any(axis=0))[0]
    for col in col_hits:
        rows = np.where(filled[:, col] > 0)[0]
        if rows.size >= 2:
            filled[rows[0] : rows[-1] + 1, col] = 1

    return filled


def refine_player_mask(
    mask: np.ndarray,
    bbox: tuple | None = None,
    frame_shape: tuple | None = None,
) -> np.ndarray:
    """
    Tighten and stabilize a single player mask.
    """
    refined = (mask > 0).astype(np.uint8)
    if refined.ndim != 2 or np.count_nonzero(refined) == 0:
        return refined

    if bbox is not None and frame_shape is not None:
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        pad_x = max(3, int((x2 - x1) * 0.05))
        pad_y = max(3, int((y2 - y1) * 0.05))

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame_w, x2 + pad_x)
        y2 = min(frame_h, y2 + pad_y)

        roi = refined[y1:y2, x1:x2]
        roi = _span_fill(roi)

        if hasattr(cv2, "getStructuringElement") and hasattr(cv2, "morphologyEx"):
            kernel_w = _odd(max(3, roi.shape[1] // 64))
            kernel_h = _odd(max(3, roi.shape[0] // 64))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
            roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

        refined[y1:y2, x1:x2] = roi
    else:
        refined = _span_fill(refined)
        if hasattr(cv2, "getStructuringElement") and hasattr(cv2, "morphologyEx"):
            kernel_w = _odd(max(3, refined.shape[1] // 80))
            kernel_h = _odd(max(3, refined.shape[0] // 80))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)

    return (refined > 0).astype(np.uint8)


def stabilize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Conservative mask cleanup for jitter reduction.
    """
    mask = (mask > 0).astype(np.uint8)
    if np.count_nonzero(mask) == 0:
        return mask

    mask = _span_fill(mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return (mask > 0).astype(np.uint8)


def create_player_removal_mask(
    frame_shape,
    boxes,
    masks,
    selected_indices,
    auxiliary_masks=None,
):
    """
    Merge selected player masks into one removal mask.

    Priority:
    1. SAM / detector masks
    2. auxiliary masks
    3. tiny bbox fallback if mask is degenerate
    """
    h, w = frame_shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)

    if boxes is None or len(boxes) == 0 or not selected_indices:
        return final_mask

    for i in selected_indices:
        if i >= len(boxes):
            continue

        x1, y1, x2, y2 = map(int, boxes[i][:4])
        player_mask = np.zeros((h, w), dtype=np.uint8)

        if masks is not None and i < len(masks):
            m = masks[i]
            if m is not None:
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                player_mask = np.maximum(player_mask, (m > 0).astype(np.uint8))

        if auxiliary_masks is not None and i < len(auxiliary_masks):
            dm = auxiliary_masks[i]
            if dm is not None:
                if dm.shape != (h, w):
                    dm = cv2.resize(dm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                player_mask = np.maximum(player_mask, (dm > 0).astype(np.uint8))

        player_mask = refine_player_mask(
            player_mask,
            bbox=(x1, y1, x2, y2),
            frame_shape=frame_shape,
        )

        if np.count_nonzero(player_mask) < 30:
            pad = 3
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
            player_mask[y1p:y2p, x1p:x2p] = 1

        final_mask = np.maximum(final_mask, player_mask)

    return final_mask


class TemporalMaskSmoother:
    """
    Small mask-only temporal smoother.
    """

    def __init__(self, history=3):
        self.history = deque(maxlen=history)

    def smooth(self, mask):
        mask = (mask > 0).astype(np.uint8)
        self.history.append(mask)

        stacked = np.stack(list(self.history), axis=0).astype(np.float32)
        avg_mask = np.mean(stacked, axis=0)

        smoothed = (avg_mask > 0.45).astype(np.uint8)

        return smoothed


def draw_selected_players(frame, boxes, selected_indices):
    output = frame.copy()

    for i in selected_indices:
        if i >= len(boxes):
            continue

        x1, y1, x2, y2 = map(int, boxes[i][:4])
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            output,
            "REMOVE",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return output