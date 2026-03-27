import cv2
import numpy as np
import torch
from PIL import Image


class SDInpainter:

    def __init__(
        self,
        target_size=512,
        pad_px=48,
    ):
        self.target_size = target_size
        self.pad_px = pad_px

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enabled = False

        if self.device != "cuda":
            print("SD disabled (no CUDA)")
            return

        try:
            from diffusers import StableDiffusionControlNetInpaintPipeline
            from diffusers import ControlNetModel
        except Exception as e:
            print("diffusers/controlnet not available:", e)
            return

        print("Loading ControlNet...")

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(self.device)

        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)

        self.enabled = True
        print("ControlNet SD ready")

    # -------------------------

    def _bbox(self, mask, shape):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        h, w = shape[:2]

        x1 = max(0, xs.min() - self.pad_px)
        y1 = max(0, ys.min() - self.pad_px)
        x2 = min(w, xs.max() + self.pad_px)
        y2 = min(h, ys.max() + self.pad_px)

        return int(x1), int(y1), int(x2), int(y2)

    # -------------------------

    def inpaint(self, frame, mask):

        if not self.enabled:
            return None

        mask = (mask > 0).astype(np.uint8)

        bbox = self._bbox(mask, frame.shape)
        if bbox is None:
            return frame

        x1, y1, x2, y2 = bbox

        crop = frame[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        if crop.size == 0:
            return frame

        h, w = crop.shape[:2]

        # 🔥 CONTROLNET INPUT (edges)
        edges = cv2.Canny(crop, 100, 200)

        edges = cv2.resize(edges, (self.target_size, self.target_size))
        edges = np.stack([edges]*3, axis=-1)

        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_size, self.target_size))

        mask_resized = cv2.resize(mask_crop, (self.target_size, self.target_size))

        image = Image.fromarray(img)
        mask_img = Image.fromarray((mask_resized * 255).astype(np.uint8))
        control = Image.fromarray(edges.astype(np.uint8))

        result = self.pipe(
            prompt="football pitch, clean grass, sharp field lines, realistic broadcast",
            negative_prompt="players, people, distortion, broken lines, blur, artifacts",
            image=image,
            mask_image=mask_img,
            control_image=control,
            guidance_scale=6.5,
            num_inference_steps=22,
        ).images[0]

        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (w, h))

        output = frame.copy()
        output[y1:y2, x1:x2] = result

        return output