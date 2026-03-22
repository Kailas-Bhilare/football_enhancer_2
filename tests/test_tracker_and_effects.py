import sys
import types

import numpy as np
import pytest

sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        GaussianBlur=lambda image, kernel, sigma: image,
        dilate=lambda image, kernel, iterations=1: image,
        morphologyEx=lambda image, op, kernel, iterations=1: image,
        getStructuringElement=lambda shape, ksize: np.ones(ksize, dtype=np.uint8),
        MORPH_CLOSE=0,
        MORPH_ELLIPSE=0,
        rectangle=lambda *args, **kwargs: None,
        putText=lambda *args, **kwargs: None,
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=0,
    ),
)

from processing.effects import (
    MotionCompensatedBackgroundReconstructor,
    TemporalMaskSmoother,
    TemporalRemovalComposer,
    create_player_removal_mask,
    refine_player_mask,
)
from processing.tracker import PlayerTracker


def test_tracker_keeps_new_tracks_alive_after_creation():
    tracker = PlayerTracker(iou_threshold=0.9, max_lost_frames=1)

    _, tracked_boxes = tracker.update([
        np.array([0, 0, 10, 10]),
        np.array([20, 20, 30, 30]),
    ])

    assert len(tracked_boxes) == 2
    assert len(tracker.tracked_players) == 2
    assert all(player["lost"] == 0 for player in tracker.tracked_players.values())


def test_tracker_clears_id_mapping_when_no_detections():
    tracker = PlayerTracker(max_lost_frames=2)

    tracker.update([np.array([0, 0, 10, 10])])
    tracker.update([])

    assert tracker.id_mapping == {}


def test_reconstructor_update_preserves_clean_background():
    reconstructor = MotionCompensatedBackgroundReconstructor()

    frame = np.full((3, 3, 3), 10, dtype=np.uint8)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 1
    output = frame.copy()
    output[1, 1] = 200

    reconstructor.update(frame, output, mask)

    assert reconstructor.background is not None
    np.testing.assert_array_equal(reconstructor.background[0, 0], frame[0, 0])
    np.testing.assert_array_equal(reconstructor.background[1, 1], output[1, 1])


def test_reconstructor_compensates_for_camera_translation():
    import cv2

    required_ops = (
        "cvtColor",
        "goodFeaturesToTrack",
        "calcOpticalFlowPyrLK",
        "estimateAffinePartial2D",
        "warpAffine",
    )
    if not all(hasattr(cv2, op) for op in required_ops):
        pytest.skip("OpenCV motion-estimation APIs unavailable in this environment")

    reconstructor = MotionCompensatedBackgroundReconstructor()
    rng = np.random.default_rng(7)

    frame = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    reconstructor.update(frame, frame, np.zeros((64, 64), dtype=np.uint8))

    dx, dy = 3, 2
    translated = np.zeros_like(frame)
    translated[dy:, dx:] = frame[:-dy, :-dx]

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:36, 24:36] = 1
    base = np.zeros_like(frame)

    reconstructed = reconstructor.reconstruct(translated, mask, base)

    expected_patch = translated[mask > 0].astype(np.int16)
    actual_patch = reconstructed[mask > 0].astype(np.int16)
    stale_patch = frame[mask > 0].astype(np.int16)

    aligned_error = np.mean(np.abs(actual_patch - expected_patch))
    stale_error = np.mean(np.abs(stale_patch - expected_patch))

    assert aligned_error < 3.0
    assert aligned_error < stale_error * 0.1


def test_temporal_composer_keeps_masked_region_bright():
    composer = TemporalRemovalComposer(blend_alpha=0.2, feather_radius=3)

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    first_output = np.full((2, 2, 3), 200, dtype=np.uint8)
    second_output = np.full((2, 2, 3), 100, dtype=np.uint8)
    mask = np.ones((2, 2), dtype=np.uint8)

    composer.compose(frame, first_output, mask)
    blended = composer.compose(frame, second_output, mask)

    expected = np.full((2, 2, 3), 120, dtype=np.uint8)
    np.testing.assert_array_equal(blended, expected)


def test_refine_player_mask_fills_internal_gaps():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[1:6, 2] = 1
    mask[1:6, 4] = 1
    mask[2, 1:6] = 1
    mask[4, 1:6] = 1

    refined = refine_player_mask(mask, bbox=(1, 1, 5, 5), frame_shape=mask.shape)

    assert refined[3, 3] == 1


def test_create_player_removal_mask_merges_fragmented_masks():
    boxes = np.array([[1, 1, 7, 7]], dtype=np.float32)
    sam_masks = np.zeros((1, 9, 9), dtype=np.uint8)
    detector_masks = np.zeros((1, 9, 9), dtype=np.uint8)

    sam_masks[0, 2:7, 3] = 1
    sam_masks[0, 2:7, 5] = 1
    detector_masks[0, 3, 2:7] = 1
    detector_masks[0, 5, 2:7] = 1

    merged = create_player_removal_mask(
        (9, 9),
        boxes,
        sam_masks,
        selected_indices={0},
        auxiliary_masks=detector_masks,
    )

    assert merged[4, 4] == 1


def test_temporal_mask_smoother_keeps_recent_partial_mask_regions():
    smoother = TemporalMaskSmoother(history=3)

    full_mask = np.zeros((6, 6), dtype=np.uint8)
    full_mask[1:5, 1:5] = 1
    partial_mask = full_mask.copy()
    partial_mask[2:4, 2:4] = 0

    smoother.smooth(full_mask)
    stabilized = smoother.smooth(partial_mask)

    assert stabilized[2, 2] == 1
    assert stabilized[3, 3] == 1
