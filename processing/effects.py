"""
Mask generation utilities with temporal smoothing.
"""

import numpy as np
import cv2
from collections import deque


class TemporalMaskSmoother:
    """
    Smooth masks across frames to reduce flicker.
    Maintains a short history plus a decaying temporal state so the mask does
    not disappear immediately when a frame under-segments the player.
    """

    def __init__(self, history=5, threshold=0.4, decay=0.7, close_size=5):
        self.history = history
        self.threshold = threshold
        self.decay = decay
        self.close_size = close_size
        self.buffer = deque(maxlen=history)
        self.state = None

    def smooth(self, mask):
        """
        mask: binary mask (H,W)
        """

        mask = (mask > 0).astype(np.uint8)
        self.buffer.append(mask.astype(np.float32))

        averaged = np.mean(np.stack(self.buffer, axis=0), axis=0)

        if self.state is None:
            self.state = averaged
        else:
            self.state = np.maximum(averaged, self.state * self.decay)

        smoothed = (self.state >= self.threshold).astype(np.uint8)

        if self.close_size > 1:
            kernel = np.ones((self.close_size, self.close_size), np.uint8)
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)

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


def mask_from_box(box, frame_shape, pad_ratio=0.08, pad_px=6):
    """
    Build a conservative bbox support mask to cover limbs missed by segmenters.
    """

    h, w = frame_shape
    x1, y1, x2, y2 = map(int, box[:4])

    pad_x = max(int((x2 - x1) * pad_ratio), pad_px)
    pad_y = max(int((y2 - y1) * pad_ratio), pad_px)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    support = np.zeros((h, w), dtype=np.uint8)
    support[y1:y2, x1:x2] = 1

    return support


def stabilize_mask(mask, dilate_size=7, blur_size=17, close_size=9):
    """
    Apply spatial cleanup and slight expansion so removal masks cover the whole
    player silhouette more reliably.
    """

    mask = (mask > 0).astype(np.uint8)

    dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
    close_kernel = np.ones((close_size, close_size), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.dilate(mask, dilate_kernel, 1)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
    mask = (mask > 0.2).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def create_player_removal_mask(
    frame_shape,
    boxes,
    masks,
    selected_indices,
    auxiliary_masks=None,
    min_mask_ratio=0.55,
):

    h, w = frame_shape

    combined = np.zeros((h, w), dtype=np.uint8)

    if boxes is None or len(boxes) == 0 or not selected_indices:
        return combined

    for idx in selected_indices:

        if idx >= len(boxes):
            continue

        player_mask = np.zeros((h, w), dtype=np.uint8)

        for mask_source in (masks, auxiliary_masks):
            if mask_source is None or idx >= len(mask_source):
                continue

            mask = mask_source[idx]

            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            player_mask = np.maximum(player_mask, mask.astype(np.uint8))

        box_mask = mask_from_box(boxes[idx], frame_shape)
        box_area = np.count_nonzero(box_mask)
        mask_area = np.count_nonzero(player_mask)

        if box_area > 0 and mask_area < box_area * min_mask_ratio:
            player_mask = np.maximum(player_mask, box_mask)

        combined = np.maximum(combined, player_mask)

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
