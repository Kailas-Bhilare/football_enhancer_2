import cv2
import numpy as np
from collections import deque


# -----------------------------
# Mask utilities
# -----------------------------


def feather_mask(mask, ksize=21):
    mask = mask.astype(np.float32)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


def _odd(value):
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _span_fill(mask):
    """
    Fill gaps between positive pixels along rows and columns.
    Reconnects fragmented body parts inside a player silhouette.
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
    """
    Close internal gaps and expand slightly to cover partial misses.

    BUG FIX: reduced dilation from (7,7)×2 to (5,5)×1.
    Over-aggressive dilation erased valid background pixels near players,
    giving the inpainter a larger hole with less surrounding context and
    causing reconstruction artifacts at player boundaries.
    """
    mask = (mask > 0).astype(np.uint8)
    mask = _span_fill(mask)

    if hasattr(cv2, "morphologyEx") and hasattr(cv2, "getStructuringElement"):
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Modest dilation — enough to cover segmentation edges without destroying context
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

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

        if masks is not None and i < len(masks):
            m = masks[i]
            if m is not None:
                player_mask = np.maximum(player_mask, (m > 0.5).astype(np.uint8))

        if auxiliary_masks is not None and i < len(auxiliary_masks):
            dm = auxiliary_masks[i]
            if dm is not None:
                player_mask = np.maximum(player_mask, (dm > 0.4).astype(np.uint8))

        player_mask = refine_player_mask(player_mask, bbox=(x1, y1, x2, y2), frame_shape=frame_shape)

        if np.sum(player_mask) < 50:
            pad = 5
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
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
# Background reconstruction (fixed)
# -----------------------------


class MotionCompensatedBackgroundReconstructor:
    """
    Maintains a rolling buffer of background frames aligned to the current
    camera position via sparse optical-flow homography estimation.

    Key fixes vs. the original implementation
    ------------------------------------------
    1. **Multi-frame background buffer** — keeps N clean (unmasked) frames and
       blends them after warping each to the current camera pose.  A single
       previous frame is unreliable when players cover large areas.

    2. **Never bake inpainting artifacts into the background model.**
       Only pixels that are *outside* the mask (i.e. genuine background) are
       used to refresh the stored frames.  The old code wrote `output[mask]`
       back into the background, which compounded inpainting errors over time.

    3. **Homography (8-DOF) instead of affine (6-DOF)** for the camera-motion
       estimate.  Broadcast cameras pan, tilt, and zoom; affine cannot model
       zoom-induced perspective, producing misaligned background patches.

    4. **Fallback chain** when optical flow fails (too few features, heavy
       occlusion, scene cut): use the most recent clean frame at its original
       position, then fall back to OpenCV inpainting alone.  Previously the
       code returned `base_output` unchanged after silently returning `None`
       from `_warp_background`, but the caller then wrote those pixels (which
       were often just zeros or stale) into the output mask region.

    5. **Explicit validity check** on warped background pixels before using
       them — pixels that map outside the frame boundary after the warp are
       not used; the inpainted `base` is kept instead for those positions.
    """

    # How many background frames to keep in the rolling buffer
    BUFFER_SIZE = 12

    def __init__(self):
        # Each entry: {'frame': ndarray, 'mask': ndarray}
        # 'mask' marks which pixels were occluded by players (1 = occluded)
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self._prev_gray = None   # For optical-flow feature tracking

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(self, frame: np.ndarray, mask: np.ndarray, base: np.ndarray) -> np.ndarray:
        """
        Fill `mask` pixels in `frame` with background content.

        Parameters
        ----------
        frame : BGR current frame (with players still visible)
        mask  : binary (0/1) uint8 — 1 where players should be removed
        base  : OpenCV-inpainted fallback (already computed by caller)

        Returns
        -------
        output : BGR frame where mask pixels are filled from the background
        """
        if len(self._buffer) == 0:
            # No history yet — caller's inpaint is the best we can do
            return base.copy()

        warped_bg, valid_map = self._build_background_estimate(frame, mask)

        # Start with the inpainted base as a safety net
        output = base.copy()

        if warped_bg is not None:
            # Only overwrite pixels where:
            #   (a) we want to remove a player  (mask == 1)
            #   (b) the warped background pixel is valid  (valid_map == 1)
            use = (mask > 0) & (valid_map > 0)
            output[use] = warped_bg[use]

            # For mask pixels where the warp gave no valid data, feather-blend
            # the inpainted base with whatever background we have
            partial = (mask > 0) & (valid_map == 0)
            if np.any(partial):
                # base already covers this via OpenCV inpaint; leave as-is
                pass

        return output

    def update(self, frame: np.ndarray, output: np.ndarray, mask: np.ndarray):
        """
        Record a new background frame.

        BUG FIX: we store the *original frame* rather than `output`.
        The inpainted/reconstructed `output` may contain artefacts; feeding
        it back as background causes error accumulation.  Instead we store
        the raw frame and its occlusion mask so future calls can skip
        occluded regions when compositing the background estimate.
        """
        binary_mask = (mask > 0).astype(np.uint8) if mask is not None \
            else np.zeros(frame.shape[:2], dtype=np.uint8)
        self._buffer.append({
            'frame': frame.copy(),
            'mask': binary_mask,
        })
        # Update the grayscale reference used for optical flow
        self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_homography(self, src_gray: np.ndarray, dst_gray: np.ndarray,
                              combined_occlusion: np.ndarray):
        """
        Estimate the homography that maps `src_gray` → `dst_gray` using
        sparse Lucas-Kanade optical flow on background feature points.

        Returns H (3×3) or None on failure.
        """
        # Build feature-detection mask: avoid player regions in BOTH frames
        feature_mask = np.ones(src_gray.shape, dtype=np.uint8) * 255
        feature_mask[combined_occlusion > 0] = 0

        if np.count_nonzero(feature_mask) < 500:
            return None  # Too much occlusion to get reliable features

        pts_src = cv2.goodFeaturesToTrack(
            src_gray,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=7,
            mask=feature_mask,
        )
        if pts_src is None or len(pts_src) < 8:
            return None

        pts_dst, status, _ = cv2.calcOpticalFlowPyrLK(
            src_gray, dst_gray, pts_src, None,
            winSize=(21, 21), maxLevel=3,
        )
        if pts_dst is None or status is None:
            return None

        keep = status.reshape(-1) == 1
        if np.count_nonzero(keep) < 8:
            return None

        H, inlier_mask = cv2.findHomography(
            pts_src[keep], pts_dst[keep],
            cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        if H is None:
            return None

        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if inliers < 6:
            return None

        return H

    def _warp_frame(self, src: np.ndarray, H: np.ndarray,
                    target_shape) -> tuple:
        """
        Warp `src` by H and return (warped_bgr, valid_pixel_mask).
        `valid_pixel_mask` is 1 where the warp produced an in-bounds pixel.
        """
        h, w = target_shape[:2]

        # Validity map: start all-ones, warp to find which output pixels came
        # from inside the source image bounds.
        ones = np.ones((src.shape[0], src.shape[1]), dtype=np.float32)
        warped = cv2.warpPerspective(src, H, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
        valid = cv2.warpPerspective(ones, H, (w, h),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
        valid_mask = (valid > 0.5).astype(np.uint8)
        return warped, valid_mask

    def _build_background_estimate(self, current_frame: np.ndarray,
                                   current_mask: np.ndarray):
        """
        Warp all buffered background frames into the current camera pose and
        blend them into a single background estimate.

        Returns
        -------
        bg_estimate : BGR ndarray or None
        valid_map   : uint8 mask (1 where bg_estimate is valid)
        """
        h, w = current_frame.shape[:2]
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        accum = np.zeros((h, w, 3), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        # Walk the buffer from newest to oldest; newer frames get more weight
        entries = list(self._buffer)  # oldest → newest
        n = len(entries)

        for idx, entry in enumerate(reversed(entries)):  # newest first
            src_frame = entry['frame']
            src_mask = entry['mask']
            src_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)

            # Combined occlusion mask for feature matching
            combined_occ = np.maximum(src_mask, current_mask)

            H = self._estimate_homography(src_gray, curr_gray, combined_occ)

            if H is not None:
                warped, valid = self._warp_frame(src_frame, H, current_frame.shape)
            else:
                if idx == 0:
                    # Most-recent frame, no homography — use direct copy
                    # (handles near-static camera or scene cuts gracefully)
                    warped = src_frame.copy()
                    valid = np.ones((h, w), dtype=np.uint8)
                else:
                    continue  # Skip older frames if we can't align them

            # Don't use pixels that were occluded in the source frame
            # (they may themselves be filled with artefacts)
            src_mask_warped = np.zeros((h, w), dtype=np.uint8)
            if H is not None and np.any(src_mask > 0):
                src_mask_warped, _ = self._warp_frame(
                    src_mask.astype(np.float32), H, current_frame.shape
                )
                src_mask_warped = (src_mask_warped > 0.3).astype(np.uint8)
            else:
                src_mask_warped = src_mask

            use = (valid > 0) & (src_mask_warped == 0)

            # Exponential decay: newest frame has highest weight
            frame_weight = np.exp(-idx * 0.35)  # ~1.0, 0.70, 0.50, 0.36 …
            pixel_weight = use.astype(np.float32) * frame_weight

            accum += warped.astype(np.float32) * pixel_weight[..., None]
            weight_sum += pixel_weight

        no_data = weight_sum < 1e-3
        if np.all(no_data):
            return None, np.zeros((h, w), dtype=np.uint8)

        # Normalise
        safe_weight = np.where(no_data, 1.0, weight_sum)
        bg_estimate = (accum / safe_weight[..., None]).astype(np.uint8)

        valid_map = (~no_data).astype(np.uint8)
        return bg_estimate, valid_map


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
            output, f"ID {pid}", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return output
