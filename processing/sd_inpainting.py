import cv2
import torch
import numpy as np
from PIL import Image


class SDInpainter:
    def __init__(
        self,
        target_size: int = 512,
        pad_px: int = 48,
        prompt: str = (
            "empty football pitch grass with field lines, "
            "realistic stadium broadcast, no players, sharp detail"
        ),
        negative_prompt: str = (
            "players, person, athlete, blurry, smeared, "
            "duplicate lines, distorted field markings, watermark"
        ),
        seed_from_position: bool = True,
    ) -> None:
        self.target_size = target_size
        self.pad_px = pad_px
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed_from_position = seed_from_position

        self.pipe = None
        self.enabled = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device != "cuda":
            print("Stable Diffusion inpainting disabled: CUDA is not available.")
            return

        try:
            from diffusers import StableDiffusionInpaintPipeline
        except ModuleNotFoundError:
            print("diffusers is not installed; disabling Stable Diffusion inpainting.")
            return
        except Exception as exc:
            print(f"Stable Diffusion import failed ({exc}); disabling inpainting.")
            return

        print("Loading Stable Diffusion inpainting model …")
        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
            )
            self.pipe = self.pipe.to(self.device)

            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception as exc:
                    print(f"xFormers unavailable ({exc}); continuing without it.")

            self.pipe.safety_checker = None
            self.pipe.set_progress_bar_config(disable=True)
            self.enabled = True
            print("Stable Diffusion ready.")
        except Exception as exc:
            self.pipe = None
            print(f"Stable Diffusion unavailable ({exc}); disabling inpainting.")

    # ------------------------------------------------------------------

    def _mask_bbox(self, mask: np.ndarray, frame_shape: tuple):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        h, w = frame_shape[:2]
        x1 = max(0, int(xs.min()) - self.pad_px)
        y1 = max(0, int(ys.min()) - self.pad_px)
        x2 = min(w, int(xs.max()) + self.pad_px + 1)
        y2 = min(h, int(ys.max()) + self.pad_px + 1)

        side = max(x2 - x1, y2 - y1)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        x1 = max(0, x2 - side)
        y1 = max(0, y2 - side)

        return x1, y1, x2, y2

    def _feather_mask(self, mask: np.ndarray, blur_size: int = 21):
        if blur_size % 2 == 0:
            blur_size += 1
        alpha = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
        return np.clip(alpha, 0.0, 1.0)

    def _make_seed(self, mask: np.ndarray):
        if not self.seed_from_position:
            return None

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        return int(cx * 1000 + cy) % (2**31)

    # ------------------------------------------------------------------

    def inpaint(self, frame: np.ndarray, mask: np.ndarray):
        if not self.enabled or self.pipe is None:
            return None

        mask = (mask > 0).astype(np.uint8)

        bbox = self._mask_bbox(mask, frame.shape)
        if bbox is None:
            return frame.copy()

        x1, y1, x2, y2 = bbox

        frame_crop = frame[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        crop_h, crop_w = frame_crop.shape[:2]

        # Resize
        frame_rgb = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_CUBIC
        )

        mask_resized = cv2.resize(
            mask_crop,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_NEAREST
        )

        image = Image.fromarray(frame_resized)

        # Soft mask (prevents seams)
        mask_soft = cv2.GaussianBlur(mask_resized.astype(np.float32), (15, 15), 0)
        mask_img = Image.fromarray((mask_soft * 255).astype(np.uint8))

        # Safe generator
        seed = self._make_seed(mask_crop)
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None

        # Inference
        result = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=image,
            mask_image=mask_img,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=28,
            strength=0.85,
        ).images[0]

        result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result_bgr = cv2.resize(
            result_bgr,
            (crop_w, crop_h),
            interpolation=cv2.INTER_CUBIC
        )

        alpha = self._feather_mask(mask_crop, blur_size=31)[..., None]

        blended_crop = (
            frame_crop.astype(np.float32) * (1.0 - alpha)
            + result_bgr.astype(np.float32) * alpha
        )

        output = frame.copy()
        output[y1:y2, x1:x2] = np.clip(blended_crop, 0, 255).astype(np.uint8)

        return output
