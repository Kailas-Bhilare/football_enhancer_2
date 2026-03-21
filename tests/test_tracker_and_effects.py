import sys
import types

import numpy as np

sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        GaussianBlur=lambda image, kernel, sigma: image,
        dilate=lambda image, kernel, iterations=1: image,
        rectangle=lambda *args, **kwargs: None,
        putText=lambda *args, **kwargs: None,
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=0,
    ),
)

from processing.effects import MotionCompensatedBackgroundReconstructor, TemporalRemovalComposer
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
