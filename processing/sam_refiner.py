import numpy as np
import cv2


class SAMRefiner:

    def __init__(self, checkpoint="sam_vit_b_01ec64.pth"):
        self.enabled = False

        try:
            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
            sam.to("cpu")

            self.predictor = SamPredictor(sam)
            self.enabled = True

            print("SAM ready")

        except Exception as e:
            print("SAM disabled:", e)
            self.predictor = None

    # -------------------------

    def _clean_mask(self, mask):
        """
        Improve mask quality:
        - fill holes
        - remove noise
        - slightly expand edges
        """

        mask = mask.astype(np.uint8)

        # fill gaps inside player
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # remove tiny noise
        mask = cv2.medianBlur(mask, 5)

        # slight dilation to ensure full coverage
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), 1)

        return mask

    # -------------------------

    def refine(self, frame, boxes):

        if not self.enabled or boxes is None or len(boxes) == 0:
            return None

        h, w = frame.shape[:2]

        # small upscale improves SAM accuracy
        scale = 1.25
        new_w = int(w * scale)
        new_h = int(h * scale)

        frame_up = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(frame_up, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(rgb)

        masks = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            # scale box
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            x2 = int(x2 * scale)
            y2 = int(y2 * scale)

            mask, _, _ = self.predictor.predict(
                box=np.array([x1, y1, x2, y2]),
                multimask_output=False
            )

            mask = mask[0].astype(np.uint8)

            # resize back
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # clean mask
            mask = self._clean_mask(mask)

            masks.append(mask)

        return np.array(masks)