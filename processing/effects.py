import cv2
import numpy as np


# -----------------------------
# Mask utilities
# -----------------------------

def feather_mask(mask, ksize=21):
    mask = mask.astype(np.float32)
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


def stabilize_mask(mask):
    mask = (mask > 0).astype(np.uint8)

    # stronger expansion (this is what you want)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), 2)

    return mask


# -----------------------------
# Mask creation
# -----------------------------

def create_player_removal_mask(
    frame_shape,
    boxes,
    masks,
    selected_indices,
    auxiliary_masks=None,
):

    h, w = frame_shape
    final_mask = np.zeros((h, w), dtype=np.uint8)

    for i in selected_indices:

        x1, y1, x2, y2 = map(int, boxes[i])

        player_mask = np.zeros((h, w), dtype=np.uint8)

        # SAM mask (preferred)
        if masks is not None and i < len(masks):
            m = masks[i]
            if m is not None:
                player_mask = np.maximum(player_mask, (m > 0.5).astype(np.uint8))

        # detector fallback
        if auxiliary_masks is not None and i < len(auxiliary_masks):
            dm = auxiliary_masks[i]
            if dm is not None:
                player_mask = np.maximum(player_mask, (dm > 0.4).astype(np.uint8))

        # HARD bbox fallback (only if mask weak)
        if np.sum(player_mask) < 50:
            pad = 5
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)

            player_mask[y1p:y2p, x1p:x2p] = 1

        final_mask = np.maximum(final_mask, player_mask)

    return final_mask


# -----------------------------
# Temporal smoothing
# -----------------------------

class TemporalMaskSmoother:
    def __init__(self, history=5):
        self.history = []
        self.max_len = history

    def smooth(self, mask):
        self.history.append(mask)

        if len(self.history) > self.max_len:
            self.history.pop(0)

        combined = np.zeros_like(mask, dtype=np.float32)

        for i, m in enumerate(self.history):
            weight = (i + 1) / len(self.history)
            combined += m * weight

        if combined.max() > 0:
            combined /= combined.max()

        return (combined > 0.4).astype(np.uint8)


# -----------------------------
# Frame composer
# -----------------------------

class TemporalRemovalComposer:
    def __init__(self, blend_alpha=0.15, feather_radius=21):
        self.prev_output = None
        self.alpha = blend_alpha
        self.feather = feather_radius

    def compose(self, frame, output, mask):

        if self.prev_output is None:
            self.prev_output = output.copy()
            return output

        mask_f = feather_mask(mask, self.feather)[..., None]
        temporal_alpha = np.clip(mask_f * self.alpha, 0.0, 1.0)

        blended = (
            output.astype(np.float32) * (1 - temporal_alpha)
            + self.prev_output.astype(np.float32) * temporal_alpha
        )

        self.prev_output = blended.astype(np.uint8)

        return self.prev_output


# -----------------------------
# Background reconstruction
# -----------------------------

class MotionCompensatedBackgroundReconstructor:

    def __init__(self):
        self.background = None

    def reconstruct(self, frame, mask, base):

        if self.background is None:
            self.background = frame.copy()

        inv_mask = (mask == 0).astype(np.uint8)

        # update only stable regions
        self.background = np.where(
            inv_mask[..., None] == 1,
            frame,
            self.background
        )

        result = base.copy()

        # fill ONLY masked region
        result[mask > 0] = self.background[mask > 0]

        return result

    def update(self, frame, output, mask):
        if self.background is None:
            self.background = output.copy()
            return

        safe_output = output.copy()
        if mask is not None:
            safe_output[mask == 0] = frame[mask == 0]

        self.background = safe_output


# -----------------------------
# Debug / UI helper
# -----------------------------

def draw_selected_players(frame, boxes, selected_indices, id_mapping=None):

    output = frame.copy()

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = map(int, box[:4])

        pid = id_mapping[i] if id_mapping and i in id_mapping else i

        color = (0, 0, 255) if pid in selected_indices else (0, 255, 0)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            output,
            f"ID {pid}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    return output
