import cv2
import numpy as np
import torch
from PIL import Image


class SDInpainter:
    def __init__(
        self,
        target_size=448,
        pad_px=40,
        prompt="empty football pitch grass, professional stadium broadcast view, clean natural grass texture, consistent mowing pattern, sharp white field lines, realistic lighting, no players",
        negative_prompt="players, people, athlete, distorted lines, duplicate lines, warped markings, blurry, smudged, artifacts, patchy grass, noise, watermark, text",
        seed_from_position=True,
    ):
        self.target_size = target_size
        self.pad_px = pad_px
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed_from_position = seed_from_position

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enabled = False
        self.pipe = None

        if self.device != "cuda":
            print("SD disabled (no CUDA)")
            return

        try:
            from diffusers import StableDiffusionInpaintPipeline
        except Exception as exc:
            print(f"diffusers unavailable ({exc}); SD disabled")
            return

        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
            ).to(self.device)

            self.pipe.safety_checker = None
            self.pipe.set_progress_bar_config(disable=True)

            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception as exc:
                    print(f"xFormers unavailable ({exc}); continuing without it.")

            self.enabled = True
            print("SD ready")
        except Exception as exc:
            self.pipe = None
            print(f"SD load failed ({exc}); disabled")

    def _mask_bbox(self, mask, frame_shape):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        h, w = frame_shape[:2]
        x1 = max(0, int(xs.min()) - self.pad_px)
        y1 = max(0, int(ys.min()) - self.pad_px)
        x2 = min(w, int(xs.max()) + self.pad_px + 1)
        y2 = min(h, int(ys.max()) + self.pad_px + 1)

        return x1, y1, x2, y2

    def _make_seed(self, mask, bbox=None):
        if not self.seed_from_position:
            return None

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
        else:
            bw = bh = 0

        return int(
            (cx // 16) * 1000003
            + (cy // 16) * 10007
            + (bw // 16) * 101
            + (bh // 16)
        ) % (2**31)

    def _looks_bad(self, crop, result, mask_resized):
        known = mask_resized == 0
        if np.count_nonzero(known) < 100:
            return False

        diff = np.mean(
            np.abs(
                crop[known].astype(np.int16) - result[known].astype(np.int16)
            )
        )
        return diff > 35

    def inpaint(self, frame, mask):
        if not self.enabled or self.pipe is None:
            return None

        mask = (mask > 0).astype(np.uint8)

        bbox = self._mask_bbox(mask, frame.shape)
        if bbox is None:
            return frame.copy()

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        if crop.size == 0:
            return frame.copy()

        h, w = crop.shape[:2]

        mask_dilated = cv2.dilate(mask_crop, np.ones((9, 9), np.uint8), 2)

        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_CUBIC,
        )

        mask_resized = cv2.resize(
            mask_dilated,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image = Image.fromarray(img)
        mask_img = Image.fromarray((mask_resized * 255).astype(np.uint8))

        seed = self._make_seed(mask_crop, bbox=bbox)
        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        try:
            result = self.pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=image,
                mask_image=mask_img,
                generator=generator,
                guidance_scale=7.5,
                num_inference_steps=24,
                strength=0.88,
            ).images[0]
        except Exception as exc:
            print(f"SD inference failed ({exc}); using TELEA fallback.")
            fallback = cv2.inpaint(
                crop,
                (mask_dilated * 255).astype(np.uint8),
                3,
                cv2.INPAINT_TELEA,
            )
            output = frame.copy()
            output[y1:y2, x1:x2] = fallback
            return output

        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)

        mask_big = cv2.resize(
            mask_dilated,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

        if self._looks_bad(crop, result, mask_big):
            fallback = cv2.inpaint(
                crop,
                (mask_dilated * 255).astype(np.uint8),
                3,
                cv2.INPAINT_TELEA,
            )
            result = fallback

        output = frame.copy()
        output[y1:y2, x1:x2] = result

        return output