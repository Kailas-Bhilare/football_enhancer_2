"""
Mask generation utilities with temporal smoothing.
"""

import numpy as np
import cv2
from collections import deque


class TemporalMaskSmoother:
    """
    Smooth masks across frames to reduce flicker.
    Keeps last N masks and averages them.
    """

    def __init__(self, history=5):
        self.history = history
        self.buffer = deque(maxlen=history)

    def smooth(self, mask):
        """
        mask: binary mask (H,W)
        """

        self.buffer.append(mask.astype(np.float32))

        stacked = np.stack(self.buffer, axis=0)

        avg_mask = np.mean(stacked, axis=0)

        # threshold back to binary
        smoothed = (avg_mask > 0.4).astype(np.uint8)

        return smoothed


def create_player_removal_mask(
    frame_shape,
    boxes,
    masks,
    selected_indices,
):

    h, w = frame_shape

    combined = np.zeros((h, w), dtype=np.uint8)

    if masks is None or len(masks) == 0:
        return combined

    for idx in selected_indices:

        if idx >= len(masks):
            continue

        mask = masks[idx]

        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        combined = np.maximum(combined, mask.astype(np.uint8))

    return combined


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
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return output