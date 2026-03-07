import cv2
import json
import argparse
import numpy as np

from config import *
from models.detector import PlayerDetector
from processing.tracker import PlayerTracker
from processing.effects import create_player_removal_mask
from processing.sd_inpainting import SDInpainter


SD_INTERVAL = 10


def load_selection():
    with open("selection.json", "r") as f:
        data = json.load(f)
    return set(data["selected_ids"])


def smooth_mask(mask):
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask,kernel,1)
    mask = cv2.GaussianBlur(mask,(15,15),0)
    return (mask>0.2).astype(np.uint8)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="result.mp4")

    args = parser.parse_args()

    selected_ids = load_selection()

    print("Players selected:", selected_ids)

    cap = cv2.VideoCapture(args.input)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width,height)
    )

    detector = PlayerDetector(YOLO_MODEL_NAME, DETECTION_CLASSES)
    tracker = PlayerTracker()
    inpainter = SDInpainter()

    frame_idx = 0

    while True:

        ret,frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        boxes,masks = detector.detect(frame)

        if boxes is None or len(boxes)==0:
            writer.write(frame)
            continue

        tracker.update(boxes)

        selected_indices = tracker.get_detection_indices(selected_ids)

        mask = create_player_removal_mask(
            frame.shape[:2],
            boxes,
            masks,
            selected_indices
        )

        mask = smooth_mask(mask)

        if np.sum(mask)>0:

            mask255 = (mask*255).astype(np.uint8)

            # fast inpainting
            output = cv2.inpaint(
                frame,
                mask255,
                3,
                cv2.INPAINT_TELEA
            )

            # occasional diffusion enhancement
            if frame_idx % SD_INTERVAL == 0:
                output = inpainter.inpaint(output, mask)

        else:
            output = frame

        writer.write(output.astype(np.uint8))

        percent = frame_idx/total*100
        print(f"\rProcessing {percent:5.1f}%  Frame {frame_idx}/{total}",end="")

    cap.release()
    writer.release()

    print("\nSaved:",args.output)


if __name__=="__main__":
    main()