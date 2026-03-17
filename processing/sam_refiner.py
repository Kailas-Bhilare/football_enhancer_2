import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


class SAMRefiner:

    def __init__(self, model_type="vit_b", checkpoint="sam_vit_b_01ec64.pth"):

        print("Loading SAM (CPU mode)...")

        device = "cpu"   # force CPU to avoid CUDA OOM

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)

        self.predictor = SamPredictor(sam)

        print("SAM ready (CPU, optimized)")

    def refine(self, frame, boxes):
        """
        frame: BGR image
        boxes: [N,4]
        returns: masks [N, H, W]
        """

        if boxes is None or len(boxes) == 0:
            return None

        orig_h, orig_w = frame.shape[:2]

        # --- resize for efficiency ---
        max_dim = 512
        scale = max_dim / max(orig_h, orig_w)

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        frame_small = cv2.resize(frame, (new_w, new_h))

        # SAM expects RGB
        frame_small_rgb = frame_small[:, :, ::-1]

        self.predictor.set_image(frame_small_rgb)

        masks = []

        for box in boxes:

            x1, y1, x2, y2 = map(int, box[:4])

            # scale box
            input_box = np.array([
                int(x1 * scale),
                int(y1 * scale),
                int(x2 * scale),
                int(y2 * scale)
            ])

            mask, _, _ = self.predictor.predict(
                box=input_box,
                multimask_output=False
            )

            mask_small = mask[0].astype(np.uint8)

            # resize back to original resolution
            mask_full = cv2.resize(
                mask_small,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

            masks.append(mask_full)

        if len(masks) == 0:
            return None

        return np.array(masks)