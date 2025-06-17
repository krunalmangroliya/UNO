import os
import tempfile
from pathlib import Path
from PIL import Image
import torch

from cog import BasePredictor, Input, Path as CogPath

from uno.flux.pipeline import UNOPipeline, preprocess_ref

class Predictor(BasePredictor):
    """
    A predictor for the UNO image generation model.
    """
    def setup(self):
        """
        Loads the UNO model into memory for efficient prediction.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # NOTE: The model_type is hardcoded to "flux-dev". You may need to
        # adjust this based on the specific model you have bundled.
        self.pipeline = UNOPipeline(
            model_type="flux-dev",
            device=self.device,
            offload=False,
            only_lora=True,
            lora_rank=512
        )

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for image generation."
        ),
        image_ref1: CogPath = Input(
            description="Reference image 1 (optional).",
            default=None,
        ),
        image_ref2: CogPath = Input(
            description="Reference image 2 (optional).",
            default=None,
        ),
        image_ref3: CogPath = Input(
            description="Reference image 3 (optional).",
            default=None,
        ),
        image_ref4: CogPath = Input(
            description="Reference image 4 (optional).",
            default=None,
        ),
        width: int = Input(
            description="Width of the generated image.",
            default=1024,
        ),
        height: int = Input(
            description="Height of the generated image.",
            default=1024,
        ),
        num_steps: int = Input(
            description="Number of diffusion steps.",
            default=25,
            ge=1,
            le=100
        ),
        guidance: float = Input(
            description="Guidance scale. Higher values adhere more closely to the prompt.",
            default=4.0,
            ge=0.0
        ),
        seed: int = Input(
            description="Seed for random number generation. Use -1 for a random seed.",
            default=-1,
        ),
        ref_size: int = Input(
            description="Reference image size for preprocessing. Use -1 for default (512 for 1 ref, 320 for multiple).",
            default=-1,
        ),
        pe: str = Input(
            description="Positional encoding type.",
            default='d',
            choices=['d', 'h', 'w', 'o']
        )
    ) -> CogPath:
        """
        Runs a single prediction on the UNO model.
        """
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        # Collect and open all provided reference images
        ref_imgs_pil = []
        for image_ref in [image_ref1, image_ref2, image_ref3, image_ref4]:
            if image_ref:
                try:
                    ref_imgs_pil.append(Image.open(str(image_ref)))
                except Exception as e:
                    raise ValueError(f"Could not open reference image: {image_ref}. Error: {e}")

        # Preprocess reference images if any were provided
        processed_ref_imgs = []
        if ref_imgs_pil:
            current_ref_size = ref_size
            if current_ref_size == -1:
                current_ref_size = 512 if len(ref_imgs_pil) == 1 else 320

            for img_pil in ref_imgs_pil:
                processed_ref_imgs.append(preprocess_ref(img_pil, current_ref_size))

        # Generate the image
        image_gen = self.pipeline(
            prompt=prompt,
            width=width,
            height=height,
            guidance=guidance,
            num_steps=num_steps,
            seed=seed,
            ref_imgs=processed_ref_imgs if processed_ref_imgs else None,
            pe=pe,
        )

        # Save the generated image to a temporary file
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        image_gen.save(str(out_path))

        return CogPath(out_path)