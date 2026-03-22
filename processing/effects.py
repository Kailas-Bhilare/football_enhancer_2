"""
processing/effects.py

Background reconstruction and mask utilities for player removal.

Architecture
------------
MotionCompensatedBackgroundReconstructor maintains a rolling buffer of raw
(unprocessed) frames alongside their player-occlusion masks.  On each frame:

  1. Every buffered frame is aligned to the current camera pose via sparse
     Lucas-Kanade optical flow + RANSAC homography (8-DOF, handles pan/tilt/zoom).
  2. Occluded pixels in each buffered frame are excluded from contributing.
  3. Valid pixels are blended with exponential time-decay weights so that
     the most recent clean observation wins.
  4. Pixels for which no buffered frame has usable data fall back to
     OpenCV TELEA inpainting as the last resort.
  5. When Stable Diffusion produces a clean inpainted region, that result is
     re-inserted into the buffer as a "virtual background frame" (mask=zeros).
     Future frames then receive it through the same homography-alignment path,
     so SD results are automatically camera-corrected without a separate cache.

Key design decisions
--------------------
* Raw frames (not outputs) are stored — inpainting artefacts never feed back.
* Homography instead of affine — models zoom correctly.
* Explicit per-pixel validity maps — black warp-border pixels are never used.
* Color-sampled boundary fill before TELEA — reduces smear on large holes.
* TemporalRemovalComposer alpha reduced to 0.15 — avoids ghost trails.
* stabilize_mask dilation reduced to (5,5)×1 — preserves surrounding context.
"""

import cv2
import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


def feather_mask(mask: np.ndarray, ksize: int = 21) -> np.ndarray:
    """Gaussian-blur a binary mask to produce a soft alpha channel."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0)


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _span_fill(mask: np.ndarray) -> np.ndarray:
    """
    Fill horizontal and vertical gaps between foreground pixels.
    Reconnects fragmented limbs inside a player silhouette so the removal
    mask does not have holes that let player pixels bleed through.
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


def refine_player_mask(
    mask: np.ndarray,
    bbox: tuple = None,
    frame_shape: tuple = None,
) -> np.ndarray:
    """
    Close internal mask gaps and lightly pad within the player bounding box.
    """
    refined = (mask > 0).astype(np.uint8)
    if refined.ndim != 2 or np.count_nonzero(refined) == 0:
        return refined

    if bbox is not None and frame_shape is not None:
        frame_h, frame_w = frame_shape
        x1, y1, x2, y2 = map(int, bbox)
        pad_x = max(6, int((x2 - x1) * 0.10))
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

    kernel_w = _odd(max(3, refined.shape[1] // 80))
    kernel_h = _odd(max(3, refined.shape[0] // 80))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)

    return (refined > 0).astype(np.uint8)


def stabilize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Close internal gaps and expand slightly to cover segmentation edge noise.

    Dilation is intentionally modest — (5,5)×1 instead of the previous
    (7,7)×2 — so that valid background pixels immediately surrounding a player
    are preserved.  Those pixels are the closest context for inpainting and
    losing them was the primary cause of boundary artefacts.
    """
    mask = (mask > 0).astype(np.uint8)
    mask = _span_fill(mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    return (mask > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mask creation
# ---------------------------------------------------------------------------


def create_player_removal_mask(
    frame_shape: tuple,
    boxes: np.ndarray,
    masks,
    selected_indices,
    auxiliary_masks=None,
) -> np.ndarray:
    """
    Merge per-player segmentation masks into one removal mask.

    Priority: SAM mask > YOLO detector mask > bbox rectangle fallback.
    """
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

        player_mask = refine_player_mask(
            player_mask, bbox=(x1, y1, x2, y2), frame_shape=frame_shape
        )

        # Bbox fallback if segmentation is degenerate
        if np.sum(player_mask) < 50:
            pad = 5
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
            player_mask[y1p:y2p, x1p:x2p] = 1

        final_mask = np.maximum(final_mask, player_mask)

    return final_mask


# ---------------------------------------------------------------------------
# Temporal mask smoothing
# ---------------------------------------------------------------------------


class TemporalMaskSmoother:
    """
    Weighted temporal union of recent masks.

    Prevents mask jitter (flickering edges) without delaying mask shrinkage
    when a player moves out of frame.
    """

    def __init__(self, history: int = 5):
        self.history: list = []
        self.max_len = history

    def smooth(self, mask: np.ndarray) -> np.ndarray:
        mask = (mask > 0).astype(np.uint8)
        self.history.append(mask)
        if len(self.history) > self.max_len:
            self.history.pop(0)

        combined = np.zeros_like(mask, dtype=np.float32)
        for i, m in enumerate(self.history):
            weight = (i + 1) / len(self.history)
            combined += m.astype(np.float32) * weight

        if combined.max() > 0:
            combined /= combined.max()

        recent_union = np.maximum.reduce(self.history[-min(3, len(self.history)):])
        stable = (combined > 0.45).astype(np.uint8)
        recovered = (combined > 0.20).astype(np.uint8) * recent_union
        smoothed = np.maximum(stable, recovered).astype(np.uint8)

        return refine_player_mask(smoothed)


# ---------------------------------------------------------------------------
# Temporal frame composer
# ---------------------------------------------------------------------------


class TemporalRemovalComposer:
    """
    Blend the current inpainted output with the previous output inside the
    mask region to reduce per-frame flicker.

    alpha=0.15 is intentionally low — just enough to smooth single-frame
    glitches without smearing moving players' traces.
    """

    def __init__(self, blend_alpha: float = 0.15, feather_radius: int = 21):
        self.prev_output: np.ndarray = None
        self.alpha = blend_alpha
        self.feather = feather_radius

    def compose(
        self, frame: np.ndarray, output: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        if self.prev_output is None:
            self.prev_output = output.copy()
            return output

        mask_f = feather_mask(mask, self.feather)[..., None]
        temporal_alpha = np.clip(mask_f * self.alpha, 0.0, 1.0)

        blended = (
            output.astype(np.float32) * (1.0 - temporal_alpha)
            + self.prev_output.astype(np.float32) * temporal_alpha
        )

        self.prev_output = blended.astype(np.uint8)
        return self.prev_output


# ---------------------------------------------------------------------------
# Background reconstruction
# ---------------------------------------------------------------------------


def _color_sample_fill(
    frame: np.ndarray, mask: np.ndarray, base: np.ndarray
) -> np.ndarray:
    """
    For pixels where background reconstruction has no data (deep occlusion),
    sample the dominant colour from the unmasked boundary ring and use it to
    seed the inpainted base.  This prevents TELEA smearing bright jersey
    colours into the grass region on large holes.

    The result is still blended with `base` so texture structure is preserved;
    only the colour balance is corrected.
    """
    if np.sum(mask) == 0:
        return base

    # Dilate mask to get a ring of known-good border pixels
    ring_outer = cv2.dilate(mask, np.ones((21, 21), np.uint8), iterations=1)
    ring = (ring_outer > 0) & (mask == 0)

    if np.sum(ring) < 10:
        return base

    # Median colour of the border ring (robust to player-adjacent pixels)
    border_pixels = frame[ring]
    median_colour = np.median(border_pixels, axis=0)  # shape (3,)

    # Create a smooth colour fill inside the mask
    colour_fill = np.zeros_like(frame, dtype=np.float32)
    colour_fill[mask > 0] = median_colour

    # Blend: 30% colour correction + 70% TELEA structure
    alpha_colour = 0.30
    result = base.copy().astype(np.float32)
    m3 = (mask > 0)[..., None]
    result = np.where(
        m3,
        result * (1.0 - alpha_colour) + colour_fill * alpha_colour,
        result,
    )
    return np.clip(result, 0, 255).astype(np.uint8)


class MotionCompensatedBackgroundReconstructor:
    """
    Rolling-buffer background reconstructor with homography alignment.

    Buffer entries
    --------------
    Each entry is a dict:
        'frame' : np.ndarray  — raw BGR frame as captured (no inpainting)
        'mask'  : np.ndarray  — uint8 binary; 1 where players were present

    SD injection
    ------------
    When Stable Diffusion produces a clean inpainted region, the caller can
    call inject_clean_frame(sd_output) to insert it into the buffer with an
    all-zeros mask.  Future reconstruct() calls will warp it to the current
    camera pose via the same homography path, giving temporally consistent
    SD results without a separate cache or any stale-pixel blending.

    Reconstruction pipeline per frame
    ----------------------------------
    1. For each buffered frame (newest → oldest):
       a. Estimate homography src→current via LK optical flow + RANSAC.
       b. Warp frame and its occlusion mask into current camera space.
       c. Compute per-pixel weight = frame_weight * (valid & ~occluded).
       d. Accumulate into weighted sum.
    2. Normalise accumulator → background estimate + validity map.
    3. Fill mask region:
       - valid background pixels  →  background estimate
       - zero-weight pixels       →  colour-corrected TELEA fallback (base)
    """

    # Rolling buffer size.  12 frames at 30 fps ≈ 0.4 s of history.
    # Large enough to survive a player standing still for several frames,
    # small enough that a panning camera does not accumulate stale alignment.
    BUFFER_SIZE = 16

    def __init__(self) -> None:
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def background(self) -> np.ndarray | None:
        """Compatibility shim — returns the most recently buffered frame."""
        if not self._buffer:
            return None
        return self._buffer[-1]["frame"]

    def reconstruct(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        base: np.ndarray,
    ) -> np.ndarray:
        """
        Fill mask pixels in *frame* from the buffer.

        Parameters
        ----------
        frame : current BGR frame (players visible, used for homography)
        mask  : binary uint8 — 1 where players should be erased
        base  : OpenCV-inpainted fallback for pixels with no buffer data

        Returns
        -------
        BGR frame with mask region filled from aligned background history.
        """
        if not self._buffer:
            return _color_sample_fill(frame, mask, base)

        bg_estimate, valid_map = self._build_background_estimate(frame, mask)

        # Start from TELEA base; overwrite only where buffer gives valid data
        output = _color_sample_fill(frame, mask, base)

        if bg_estimate is not None:
            use = (mask > 0) & (valid_map > 0)
            if np.any(use):
                # Feather the transition at the boundary for a smooth blend
                alpha = feather_mask(use.astype(np.uint8), ksize=5)[..., None]
                output = np.where(
                    (mask > 0)[..., None],
                    bg_estimate.astype(np.float32) * alpha
                    + output.astype(np.float32) * (1.0 - alpha),
                    output.astype(np.float32),
                ).astype(np.uint8)

        return output

    def update(
        self,
        frame: np.ndarray,
        output: np.ndarray,  # kept for API compatibility; NOT stored
        mask: np.ndarray,
    ) -> None:
        """
        Record the raw frame and its occlusion mask.

        IMPORTANT: `output` (the inpainted result) is deliberately ignored.
        Storing processed output would feed inpainting artefacts back into
        future reconstructions, causing error accumulation over time.
        Only the raw camera frame is buffered.
        """
        binary_mask = (
            (mask > 0).astype(np.uint8)
            if mask is not None
            else np.zeros(frame.shape[:2], dtype=np.uint8)
        )
        self._buffer.append({"frame": frame.copy(), "mask": binary_mask})

    def inject_clean_frame(self, clean_bgr: np.ndarray) -> None:
        """
        Insert a fully clean background frame (e.g. SD inpaint output) into
        the buffer with an all-zeros mask.

        This frame will be warped to future camera poses by the same homography
        path as regular frames, so its pixels are reused in a camera-consistent
        manner without any extra caching logic in the caller.

        Call this immediately after a successful SD inpaint.
        """
        h, w = clean_bgr.shape[:2]
        self._buffer.append(
            {
                "frame": clean_bgr.copy(),
                "mask": np.zeros((h, w), dtype=np.uint8),  # all pixels clean
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_homography(
        self,
        src_gray: np.ndarray,
        dst_gray: np.ndarray,
        combined_occlusion: np.ndarray,
    ) -> np.ndarray | None:
        """
        Estimate 8-DOF homography src → dst via sparse LK optical flow + RANSAC.

        Features are detected only in regions free of player occlusion in both
        src and dst.  Homography (vs. affine) correctly handles camera zoom.

        Returns H (3×3 float64) or None on failure.
        """
        feature_mask = np.ones(src_gray.shape, dtype=np.uint8) * 255
        feature_mask[combined_occlusion > 0] = 0

        # Need a reasonable number of pixels free of occlusion
        if np.count_nonzero(feature_mask) < 1000:
            return None

        pts_src = cv2.goodFeaturesToTrack(
            src_gray,
            maxCorners=600,
            qualityLevel=0.01,
            minDistance=7,
            mask=feature_mask,
        )
        if pts_src is None or len(pts_src) < 8:
            return None

        pts_dst, status, _ = cv2.calcOpticalFlowPyrLK(
            src_gray,
            dst_gray,
            pts_src,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if pts_dst is None or status is None:
            return None

        keep = status.reshape(-1) == 1
        if np.count_nonzero(keep) < 8:
            return None

        H, inlier_mask = cv2.findHomography(
            pts_src[keep],
            pts_dst[keep],
            cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if H is None:
            return None

        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if inliers < 6:
            return None

        return H

    def _warp_with_validity(
        self,
        src: np.ndarray,
        H: np.ndarray,
        target_shape: tuple,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Warp *src* by homography H into *target_shape* space.

        Returns
        -------
        warped    : BGR ndarray in target space
        valid_map : uint8 mask; 1 where source pixels existed (no border fill)
        """
        h, w = target_shape[:2]
        src_ones = np.ones(src.shape[:2], dtype=np.float32)

        warped = cv2.warpPerspective(
            src, H, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_raw = cv2.warpPerspective(
            src_ones, H, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_map = (valid_raw > 0.5).astype(np.uint8)
        return warped, valid_map

    def _build_background_estimate(
        self,
        current_frame: np.ndarray,
        current_mask: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Warp all buffered frames into the current camera pose and blend them.

        Each buffered frame's contribution to a pixel is gated by:
          - homography validity (pixel maps from inside source bounds)
          - source occlusion mask (pixel was not covered by a player there either)

        Exponential time-decay weights (newest ≈ 1.0, oldest ≈ e^{-0.35*N})
        ensure recent clean observations dominate.

        Returns
        -------
        bg_estimate : (H, W, 3) uint8 or None
        valid_map   : (H, W) uint8, 1 where bg_estimate has data
        """
        h, w = current_frame.shape[:2]
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        accum = np.zeros((h, w, 3), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        # Iterate newest-first so idx==0 is most recent
        entries = list(self._buffer)  # oldest … newest
        for idx, entry in enumerate(reversed(entries)):
            src_frame: np.ndarray = entry["frame"]
            src_occ: np.ndarray = entry["mask"]

            src_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
            combined_occ = np.maximum(src_occ, current_mask)

            H = self._estimate_homography(src_gray, curr_gray, combined_occ)

            if H is not None:
                warped, hw_valid = self._warp_with_validity(
                    src_frame, H, current_frame.shape
                )
                # Warp the source occlusion mask into current space
                occ_warped_f, _ = self._warp_with_validity(
                    src_occ.astype(np.float32), H, current_frame.shape
                )
                occ_warped = (occ_warped_f > 0.3).astype(np.uint8)
            else:
                if idx == 0:
                    # Most-recent frame with no usable homography (scene cut,
                    # near-static camera, etc.) — use it unwarped as a
                    # last-resort anchor.
                    warped = src_frame.copy()
                    hw_valid = np.ones((h, w), dtype=np.uint8)
                    occ_warped = src_occ
                else:
                    # Older frame + no alignment = unreliable; skip it.
                    continue

            # Only use pixels that are:
            #   a) within source image bounds after the warp
            #   b) NOT occluded in the source frame
            use = (hw_valid > 0) & (occ_warped == 0)

            # Exponential decay: idx 0 → weight≈1.0, idx 1 → 0.70, idx 2 → 0.50 …
            frame_weight = float(np.exp(-idx * 0.35))
            pixel_weight = use.astype(np.float32) * frame_weight

            accum += warped.astype(np.float32) * pixel_weight[..., None]
            weight_sum += pixel_weight

        no_data = weight_sum < 1e-3
        if np.all(no_data):
            return None, np.zeros((h, w), dtype=np.uint8)

        safe_w = np.where(no_data, 1.0, weight_sum)
        bg_estimate = np.clip(accum / safe_w[..., None], 0, 255).astype(np.uint8)
        valid_map = (~no_data).astype(np.uint8)

        return bg_estimate, valid_map


# ---------------------------------------------------------------------------
# Debug / UI helper
# ---------------------------------------------------------------------------


def draw_selected_players(
    frame: np.ndarray,
    boxes: np.ndarray,
    selected_indices,
    id_mapping: dict = None,
) -> np.ndarray:
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
            cv2.LINE_AA,
        )
    return output
