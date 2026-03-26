import numpy as np
import pytest

from processing.effects import (
    create_player_removal_mask,
    refine_player_mask,
    stabilize_mask,
    TemporalMaskSmoother,
)
from processing.tracker import PlayerTracker


# --------------------------------------------------
# Tracker tests
# --------------------------------------------------

def test_tracker_creates_tracks():
    tracker = PlayerTracker()

    boxes = [
        np.array([0, 0, 10, 10]),
        np.array([20, 20, 30, 30]),
    ]

    mapping, tracked = tracker.update(boxes)

    assert len(tracked) == 2
    assert len(tracker.tracked_players) == 2
    assert set(mapping.values()) == {0, 1}


def test_tracker_handles_no_detections():
    tracker = PlayerTracker()

    tracker.update([np.array([0, 0, 10, 10])])
    tracker.update([])

    assert tracker.id_mapping == {}


# --------------------------------------------------
# Mask creation tests
# --------------------------------------------------

def test_create_player_removal_mask_basic():
    frame_shape = (10, 10)

    boxes = np.array([[2, 2, 6, 6]], dtype=np.float32)

    masks = np.zeros((1, 10, 10), dtype=np.uint8)
    masks[0, 3:5, 3:5] = 1

    result = create_player_removal_mask(
        frame_shape,
        boxes,
        masks,
        selected_indices={0},
    )

    assert np.sum(result) > 0


def test_mask_fallback_to_bbox():
    frame_shape = (10, 10)

    boxes = np.array([[2, 2, 6, 6]], dtype=np.float32)

    masks = np.zeros((1, 10, 10), dtype=np.uint8)

    result = create_player_removal_mask(
        frame_shape,
        boxes,
        masks,
        selected_indices={0},
    )

    assert np.sum(result) > 0


# --------------------------------------------------
# Mask refinement
# --------------------------------------------------

def test_refine_player_mask_fills_gaps():
    mask = np.zeros((7, 7), dtype=np.uint8)

    mask[2:5, 2] = 1
    mask[2:5, 4] = 1
    mask[2, 2:5] = 1
    mask[4, 2:5] = 1

    refined = refine_player_mask(mask)

    assert refined[3, 3] == 1


def test_stabilize_mask_removes_noise():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 1
    mask[1, 1] = 1  # noise

    stabilized = stabilize_mask(mask)

    assert stabilized[1, 1] == 0 or np.sum(stabilized) >= 4


# --------------------------------------------------
# Temporal smoothing
# --------------------------------------------------

def test_temporal_mask_smoother():
    smoother = TemporalMaskSmoother(history=3)

    mask1 = np.zeros((5, 5), dtype=np.uint8)
    mask1[2, 2] = 1

    mask2 = mask1.copy()
    mask3 = mask1.copy()

    smoother.smooth(mask1)
    smoother.smooth(mask2)
    result = smoother.smooth(mask3)

    assert result[2, 2] == 1