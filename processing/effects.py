"""
Mask generation utilities with temporal smoothing.
"""

import numpy as np
import cv2
from collections import deque


class TemporalMaskSmoother:
    """
    Smooth masks across frames to reduce flicker.
    Keeps a short history and uses hysteresis to avoid masks popping.
    """

    def __init__(self, history=5, threshold=0.35, persistence=0.15):
        self.history = history
        self.threshold = threshold
        self.persistence = persistence
        self.buffer = deque(maxlen=history)
        self.prev_mask = None

    def smooth(self, mask):
        """
        mask: binary mask (H,W)
        """

        mask = (mask > 0).astype(np.uint8)
        self.buffer.append(mask.astype(np.float32))

        stacked = np.stack(self.buffer, axis=0)
        avg_mask = np.mean(stacked, axis=0)

        if self.prev_mask is not None:
            avg_mask = np.maximum(
                avg_mask,
                self.prev_mask.astype(np.float32) * self.persistence,
            )

        smoothed = (avg_mask >= self.threshold).astype(np.uint8)
        self.prev_mask = smoothed

        return smoothed


class TemporalRemovalComposer:
    """
    Blend newly inpainted regions with the previous result only inside the
    removal area. This stabilizes inpainting without ghosting the full frame.
    """

    def __init__(self, blend_alpha=0.35, feather_radius=21):
        self.blend_alpha = blend_alpha
        self.feather_radius = feather_radius
        self.prev_output = None
        self.prev_mask = None

    def compose(self, frame, inpainted, mask):
        mask = (mask > 0).astype(np.uint8)

        if self.prev_output is not None and self.prev_output.shape != inpainted.shape:
            self.prev_output = cv2.resize(
                self.prev_output,
                (inpainted.shape[1], inpainted.shape[0]),
            )

        if self.prev_mask is not None and self.prev_mask.shape != mask.shape:
            self.prev_mask = cv2.resize(
                self.prev_mask,
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        mask_alpha = feather_mask(mask, self.feather_radius)
        output = frame.astype(np.float32) * (1.0 - mask_alpha[..., None])
        output += inpainted.astype(np.float32) * mask_alpha[..., None]

        if self.prev_output is not None and self.prev_mask is not None:
            overlap = cv2.bitwise_and(mask, self.prev_mask)
            overlap_alpha = feather_mask(overlap, self.feather_radius)

            if np.any(overlap_alpha > 0):
                stabilized = (
                    self.prev_output.astype(np.float32) * self.blend_alpha
                    + output * (1.0 - self.blend_alpha)
                )
                output = (
                    output * (1.0 - overlap_alpha[..., None])
                    + stabilized * overlap_alpha[..., None]
                )

        final = np.clip(output, 0, 255).astype(np.uint8)
        self.prev_output = final.copy()
        self.prev_mask = mask.copy()

        return final


def feather_mask(mask, blur_size=21):
    """
    Convert a binary mask into a soft alpha matte for seam-free compositing.
    """

    mask = (mask > 0).astype(np.uint8)

    if blur_size % 2 == 0:
        blur_size += 1

    if blur_size <= 1:
        return mask.astype(np.float32)

    alpha = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
    return np.clip(alpha, 0.0, 1.0)


def stabilize_mask(mask, dilate_size=5, blur_size=15, close_size=7):
    """
    Apply light spatial cleanup before temporal smoothing.
    """

    mask = (mask > 0).astype(np.uint8)

    dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
    close_kernel = np.ones((close_size, close_size), np.uint8)

    mask = cv2.dilate(mask, dilate_kernel, 1)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
    mask = (mask > 0.25).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


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
