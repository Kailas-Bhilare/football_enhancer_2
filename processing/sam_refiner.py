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

    def refine(self, frame, boxes):

        if not self.enabled or boxes is None or len(boxes) == 0:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

        masks = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            mask, _, _ = self.predictor.predict(
                box=np.array([x1, y1, x2, y2]),
                multimask_output=False
            )

            masks.append(mask[0].astype(np.uint8))

        return np.array(masks)