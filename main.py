"""
Selection UI.

Features:
* click players to remove
* pause on click
* arrow keys for frame stepping
* player IDs displayed
"""

import cv2
import json
import argparse

from config import *
from models.detector import PlayerDetector
from processing.tracker import PlayerTracker


# --------------------------------------------------
# Constants (FIXED)
# --------------------------------------------------
WINDOW_NAME = "Player Selection"

KEY_QUIT = ord("q")
KEY_SAVE = ord("s")
KEY_SPACE = ord(" ")

KEY_LEFT = 81   # ←
KEY_RIGHT = 83  # →


# --------------------------------------------------
# State
# --------------------------------------------------
class AppState:
    def __init__(self):
        self.selected_ids = set()
        self.current_boxes = []
        self.tracker = None
        self.paused = False
        self.current_frame = None


# --------------------------------------------------
# Mouse
# --------------------------------------------------
def mouse_callback(event, x, y, flags, state):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.paused = True

        for i, box in enumerate(state.current_boxes):
            x1, y1, x2, y2 = map(int, box[:4])

            if x1 <= x <= x2 and y1 <= y <= y2:
                if i in state.tracker.id_mapping:
                    pid = state.tracker.id_mapping[i]

                    if pid not in state.selected_ids:
                        state.selected_ids.add(pid)
                        print("Selected player", pid)
                break

    elif event == cv2.EVENT_RBUTTONDOWN:
        state.paused = True

        for i, box in enumerate(state.current_boxes):
            x1, y1, x2, y2 = map(int, box[:4])

            if x1 <= x <= x2 and y1 <= y <= y2:
                if i in state.tracker.id_mapping:
                    pid = state.tracker.id_mapping[i]

                    if pid in state.selected_ids:
                        state.selected_ids.remove(pid)
                        print("Removed player", pid)
                break


# --------------------------------------------------
# Drawing
# --------------------------------------------------
def draw_boxes(frame, boxes, tracker, selected_ids):

    output = frame.copy()

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = map(int, box[:4])

        if i in tracker.id_mapping:
            pid = tracker.id_mapping[i]

            color = (0, 0, 255) if pid in selected_ids else (0, 255, 0)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                output,
                f"ID {pid}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    return output


# --------------------------------------------------
# Save
# --------------------------------------------------
def save_selection(ids):

    data = {"selected_ids": list(ids)}

    with open("selection.json", "w") as f:
        json.dump(data, f, indent=2)

    print("Saved selection.json")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print("Cannot open video")
        return

    detector = PlayerDetector(YOLO_MODEL_NAME, DETECTION_CLASSES)
    tracker = PlayerTracker()

    state = AppState()
    state.tracker = tracker

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)

    print("\nControls")
    print("Left click  → select player")
    print("Right click → unselect player")
    print("Space       → pause/play")
    print("→ arrow     → next frame")
    print("← arrow     → previous frame")
    print("S           → save selection")
    print("Q           → quit\n")

    while True:

        if not state.paused:
            ret, frame = cap.read()

            if not ret:
                print("End of video")
                break

            state.current_frame = frame

        if state.current_frame is None:
            continue

        frame = state.current_frame.copy()

        boxes, masks = detector.detect(frame)

        if boxes is not None and len(boxes) > 0:

            id_mapping, tracked_boxes = tracker.update(boxes)

            state.current_boxes = tracked_boxes

            frame = draw_boxes(
                frame,
                tracked_boxes,
                tracker,
                state.selected_ids
            )
        else:
            tracker.update([])
            state.current_boxes = []

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(30) & 0xFF

        # ----------------------------
        # Controls
        # ----------------------------

        if key == KEY_QUIT:
            break

        elif key == KEY_SPACE:
            state.paused = not state.paused

        elif key == KEY_RIGHT:
            state.paused = True
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                cap.get(cv2.CAP_PROP_POS_FRAMES) + 1
            )

        elif key == KEY_LEFT:
            state.paused = True
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(pos - 2, 0))

        elif key == KEY_SAVE:
            save_selection(state.selected_ids)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()