import torch
from cog import BasePredictor, Input, Path
from PIL import Image

from uno.flux.pipeline import UNOPipeline, preprocess_ref


class Predictor(BasePredictor):
    def setup(self):
        """Load the UNO pipeline."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = UNOPipeline(
            model_type="flux-dev",
            device=device,
            offload=False,
            only_lora=True,
            lora_rank=512,
        )

    def predict(
        self,
        prompt: str = Input(description="Prompt describing the desired image"),
        ref_image1: Path | None = Input(description="Reference image 1", default=None),
        ref_image2: Path | None = Input(description="Reference image 2", default=None),
        ref_image3: Path | None = Input(description="Reference image 3", default=None),
        ref_image4: Path | None = Input(description="Reference image 4", default=None),
        width: int = Input(default=512, description="Generation width"),
        height: int = Input(default=512, description="Generation height"),
        guidance: float = Input(default=4.0, description="Guidance scale"),
        num_steps: int = Input(default=25, description="Number of diffusion steps"),
        seed: int = Input(default=3407, description="Random seed"),
        pe: str = Input(default="d", choices=["d", "h", "w", "o"], description="PE type"),
    ) -> Path:
        """Run a prediction using UNO."""
        ref_imgs = []
        for path in [ref_image1, ref_image2, ref_image3, ref_image4]:
            if path is not None:
                img = Image.open(path)
                ref_imgs.append(img)
        if ref_imgs:
            ref_long_side = 512 if len(ref_imgs) == 1 else 320
            ref_imgs = [preprocess_ref(img, ref_long_side) for img in ref_imgs]

        result = self.pipeline(
            prompt=prompt,
            width=width,
            height=height,
            guidance=guidance,
            num_steps=num_steps,
            seed=seed,
            ref_imgs=ref_imgs,
            pe=pe,
        )

        out_path = Path("/tmp/out.png")
        result.save(out_path)
        return out_path
