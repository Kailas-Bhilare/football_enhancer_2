import cv2
import numpy as np


# -----------------------------
# Mask utilities
# -----------------------------


def feather_mask(mask, ksize=21):
    mask = mask.astype(np.float32)
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


def _odd(value):
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _span_fill(mask):
    """
    Fill gaps between positive pixels along rows and columns.
    This helps reconnect fragmented body parts inside a player silhouette.
    """

    filled = mask.copy().astype(np.uint8)

    if filled.ndim != 2 or np.count_nonzero(filled) == 0:
        return filled

    row_hits = np.where(filled.any(axis=1))[0]
    for row in row_hits:
        cols = np.where(filled[row] > 0)[0]
        if cols.size >= 2:
            filled[row, cols[0]:cols[-1] + 1] = 1

    col_hits = np.where(filled.any(axis=0))[0]
    for col in col_hits:
        rows = np.where(filled[:, col] > 0)[0]
        if rows.size >= 2:
            filled[rows[0]:rows[-1] + 1, col] = 1

    return filled


def refine_player_mask(mask, bbox=None, frame_shape=None):
    """
    Make a single-player mask less choppy by reconnecting fragmented regions
    and softly padding the result inside the player's bounding box.
    """

    refined = (mask > 0).astype(np.uint8)

    if refined.ndim != 2 or np.count_nonzero(refined) == 0:
        return refined

    if bbox is not None and frame_shape is not None:
        frame_h, frame_w = frame_shape
        x1, y1, x2, y2 = map(int, bbox)
        pad_x = max(6, int((x2 - x1) * 0.1))
        pad_y = max(6, int((y2 - y1) * 0.08))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame_w, x2 + pad_x)
        y2 = min(frame_h, y2 + pad_y)
        roi = refined[y1:y2, x1:x2]
        roi = _span_fill(roi)
        refined[y1:y2, x1:x2] = roi
    else:
        refined = _span_fill(refined)

    if hasattr(cv2, "morphologyEx") and hasattr(cv2, "getStructuringElement"):
        kernel_w = _odd(max(3, refined.shape[1] // 80))
        kernel_h = _odd(max(3, refined.shape[0] // 80))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)

    return (refined > 0).astype(np.uint8)


def stabilize_mask(mask):
    mask = (mask > 0).astype(np.uint8)
    mask = _span_fill(mask)

    if hasattr(cv2, "morphologyEx") and hasattr(cv2, "getStructuringElement"):
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # stronger expansion to cover partial misses around limbs and edges
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), 2)

    return (mask > 0).astype(np.uint8)


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

        # detector fallback / complement
        if auxiliary_masks is not None and i < len(auxiliary_masks):
            dm = auxiliary_masks[i]
            if dm is not None:
                player_mask = np.maximum(player_mask, (dm > 0.4).astype(np.uint8))

        player_mask = refine_player_mask(player_mask, bbox=(x1, y1, x2, y2), frame_shape=frame_shape)

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
        mask = (mask > 0).astype(np.uint8)
        self.history.append(mask)

        if len(self.history) > self.max_len:
            self.history.pop(0)

        combined = np.zeros_like(mask, dtype=np.float32)

        for i, m in enumerate(self.history):
            weight = (i + 1) / len(self.history)
            combined += m * weight

        if combined.max() > 0:
            combined /= combined.max()

        recent_union = np.maximum.reduce(self.history[-min(3, len(self.history)):])
        stable = (combined > 0.45).astype(np.uint8)
        recovered = (combined > 0.2).astype(np.uint8) * recent_union
        smoothed = np.maximum(stable, recovered).astype(np.uint8)

        return refine_player_mask(smoothed)


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
        self.prev_frame = None
        self.prev_mask = None

    def _estimate_camera_transform(self, frame, mask):
        if self.prev_frame is None or self.background is None:
            return None

        required_ops = (
            "cvtColor",
            "goodFeaturesToTrack",
            "calcOpticalFlowPyrLK",
            "estimateAffinePartial2D",
        )
        if not all(hasattr(cv2, op) for op in required_ops):
            return None

        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        valid_mask = np.ones(mask.shape, dtype=np.uint8) * 255
        valid_mask[mask > 0] = 0

        if self.prev_mask is not None:
            valid_mask[self.prev_mask > 0] = 0

        if np.count_nonzero(valid_mask) < 32:
            return None

        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=7,
            mask=valid_mask,
        )

        if prev_pts is None or len(prev_pts) < 6:
            return None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if curr_pts is None or status is None:
            return None

        keep = status.reshape(-1) == 1
        if np.count_nonzero(keep) < 6:
            return None

        matrix, _ = cv2.estimateAffinePartial2D(
            prev_pts[keep],
            curr_pts[keep],
            method=getattr(cv2, "RANSAC", 0),
            ransacReprojThreshold=3.0,
        )

        return matrix

    def _warp_background(self, frame, mask):
        if self.background is None:
            return None

        matrix = self._estimate_camera_transform(frame, mask)
        if matrix is None or not hasattr(cv2, "warpAffine"):
            return self.background.copy()

        height, width = frame.shape[:2]
        return cv2.warpAffine(
            self.background,
            matrix,
            (width, height),
            flags=getattr(cv2, "INTER_LINEAR", 1),
            borderMode=getattr(cv2, "BORDER_REPLICATE", 1),
        )

    def reconstruct(self, frame, mask, base):

        if self.background is None:
            self.background = frame.copy()
            self.prev_frame = frame.copy()
            self.prev_mask = (mask > 0).astype(np.uint8)

        aligned_background = self._warp_background(frame, mask)
        inv_mask = (mask == 0).astype(np.uint8)

        # refresh stable regions from the current frame in camera-aligned space
        refreshed_background = np.where(
            inv_mask[..., None] == 1,
            frame,
            aligned_background
        )

        result = base.copy()

        # fill only the removed player region from the aligned background estimate
        result[mask > 0] = refreshed_background[mask > 0]

        self.background = refreshed_background

        return result

    def update(self, frame, output, mask):
        if self.background is None:
            self.background = output.copy()
            self.prev_frame = frame.copy()
            self.prev_mask = None if mask is None else (mask > 0).astype(np.uint8)
            return

        safe_output = self.background.copy()
        if mask is None:
            safe_output = output.copy()
        else:
            safe_output[mask > 0] = output[mask > 0]

        self.background = safe_output
        self.prev_frame = frame.copy()
        self.prev_mask = None if mask is None else (mask > 0).astype(np.uint8)


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
