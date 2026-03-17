"""
Stable Diffusion inpainting module.

Uses fixed seed for consistent results across frames.
"""

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


class SDInpainter:

    def __init__(self):

        print("Loading Stable Diffusion inpainting...")

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
        )

        self.pipe = self.pipe.to("cuda")

        # memory optimization
        self.pipe.enable_xformers_memory_efficient_attention()

        # disable safety checker (faster)
        self.pipe.safety_checker = None

        # disable progress bar spam
        self.pipe.set_progress_bar_config(disable=True)

        # fixed seed generator (IMPORTANT)
        self.generator = torch.Generator(device="cuda").manual_seed(42)

        print("Stable Diffusion ready")

    def inpaint(self, frame, mask):
        """
        frame : BGR numpy image
        mask  : binary mask (0/1)
        """

        # BGR → RGB
        frame_rgb = frame[:, :, ::-1]

        image = Image.fromarray(frame_rgb)

        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        result = self.pipe(
            prompt="football field grass stadium",
            image=image,
            mask_image=mask_img,
            generator=self.generator,          # fixed seed
            guidance_scale=6.5,
            num_inference_steps=20,
        ).images[0]

        result = np.array(result)

        # RGB → BGR
        result = result[:, :, ::-1]

        return result