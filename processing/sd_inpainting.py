import cv2
import numpy as np
import torch
from PIL import Image


class SDInpainter:

    def __init__(
        self,
        target_size=512,
        pad_px=48,
        prompt="empty football pitch grass, professional stadium broadcast view, clean natural grass texture, consistent mowing pattern, sharp white field lines, realistic lighting, no players, no shadows of people, high detail, natural color, smooth grass surface",
        negative_prompt="players, people, athlete, distorted lines, duplicate lines, warped markings, blurry, smudged, artifacts, patchy grass, noise, watermark, text",
    ):
        self.target_size = target_size
        self.pad_px = pad_px
        self.prompt = prompt
        self.negative_prompt = negative_prompt

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enabled = False

        if self.device != "cuda":
            print("SD disabled (no CUDA)")
            return

        from diffusers import StableDiffusionInpaintPipeline

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
        ).to(self.device)

        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)

        self.enabled = True
        print("SD ready")

    # --------------------------------------------------

    def _bbox(self, mask, shape):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        h, w = shape[:2]

        x1 = max(0, xs.min() - self.pad_px)
        y1 = max(0, ys.min() - self.pad_px)
        x2 = min(w, xs.max() + self.pad_px)
        y2 = min(h, ys.max() + self.pad_px)

        return x1, y1, x2, y2

    # --------------------------------------------------

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

        h, w = crop.shape[:2]

        # 🔥 EXPAND MASK for full removal
        mask_dilated = cv2.dilate(mask_crop, np.ones((9, 9), np.uint8), 2)

        # resize
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_size, self.target_size))

        mask_resized = cv2.resize(mask_dilated, (self.target_size, self.target_size))

        image = Image.fromarray(img)
        mask_img = Image.fromarray((mask_resized * 255).astype(np.uint8))

        result = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=image,
            mask_image=mask_img,
            guidance_scale=7.5,
            num_inference_steps=28,
            strength=0.9,
        ).images[0]

        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (w, h))

        output = frame.copy()
        output[y1:y2, x1:x2] = result

        return output