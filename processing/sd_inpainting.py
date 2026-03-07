"""
Stable Diffusion inpainting module.
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

        # disable safety checker
        self.pipe.safety_checker = None

        # disable diffusion progress bars
        self.pipe.set_progress_bar_config(disable=True)

        print("Stable Diffusion ready")

    def inpaint(self, frame, mask):
        """
        frame : BGR numpy image
        mask  : binary mask where white pixels should be filled
        """

        frame_rgb = frame[:, :, ::-1]

        image = Image.fromarray(frame_rgb)

        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        result = self.pipe(
            prompt="football field grass stadium background",
            image=image,
            mask_image=mask_img,
            guidance_scale=6.5,
            num_inference_steps=20,
        ).images[0]

        result = np.array(result)

        result = result[:, :, ::-1]

        return result