import os
from pathlib import Path

import cv2
import numpy as np


class SAMRefiner:

    def __init__(self, model_type="vit_b", checkpoint="sam_vit_b_01ec64.pth"):

        self.predictor = None
        self.enabled = False
        self.model_type = model_type
        self.checkpoint = self._resolve_checkpoint_path(checkpoint)

        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ModuleNotFoundError:
            print("segment_anything is not installed; continuing with detector masks only.")
            return
        except Exception as exc:
            print(f"SAM import failed ({exc}); continuing with detector masks only.")
            return

        if self.checkpoint is None:
            print("SAM checkpoint not found; continuing with detector masks only.")
            return

        print("Loading SAM (CPU mode)...")

        device = "cpu"   # force CPU to avoid CUDA OOM

        try:
            sam = sam_model_registry[model_type](checkpoint=self.checkpoint)
            sam.to(device)
            self.predictor = SamPredictor(sam)
            self.enabled = True
            print(f"SAM ready (CPU, optimized) from {self.checkpoint}")
        except FileNotFoundError:
            print("SAM checkpoint disappeared during load; continuing with detector masks only.")
        except Exception as exc:
            print(f"SAM unavailable ({exc}); continuing with detector masks only.")

    def _resolve_checkpoint_path(self, checkpoint):
        candidates = []

        if checkpoint:
            candidates.append(Path(checkpoint))

        env_checkpoint = os.environ.get("SAM_CHECKPOINT")
        if env_checkpoint:
            candidates.append(Path(env_checkpoint))

        if checkpoint:
            candidates.extend([
                Path(__file__).resolve().parent.parent / checkpoint,
                Path.cwd() / checkpoint,
                Path.home() / checkpoint,
                Path.home() / ".cache" / "sam" / checkpoint,
                Path.home() / "checkpoints" / checkpoint,
            ])

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate)

        return None

    def refine(self, frame, boxes):
        """
        frame: BGR image
        boxes: [N,4]
        returns: masks [N, H, W]
        """

        if not self.enabled or self.predictor is None:
            return None

        if boxes is None or len(boxes) == 0:
            return None

        orig_h, orig_w = frame.shape[:2]

        # --- resize for efficiency ---
        max_dim = 512
        scale = min(1.0, max_dim / max(orig_h, orig_w))

        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        frame_small = cv2.resize(frame, (new_w, new_h)) if scale != 1.0 else frame

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
            ) if scale != 1.0 else mask_small

            masks.append(mask_full)

        if len(masks) == 0:
            return None

        return np.array(masks)
