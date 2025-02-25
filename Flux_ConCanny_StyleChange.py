import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers import FluxControlNetPipeline
from controlnet_aux import CannyDetector
from PIL import Image
import numpy as np
import os

generator = torch.Generator(device="cuda").manual_seed(87544357)

controlnet = FluxControlNetModel.from_pretrained(
  "./models/flux-controlnet-canny-diffusers",
  torch_dtype=torch.bfloat16,
  use_safetensors=True,
)
pipe = FluxControlNetPipeline.from_pretrained(
  "./models/FLUX.1-dev",
  controlnet=controlnet,
  torch_dtype=torch.bfloat16
)
pipe.to("cuda")

working_dir = "inputs"
for image_path in sorted([os.path.join(working_dir, f) for f in os.listdir(working_dir) if f.endswith((".png", ".jpg", ".jpeg"))]):
    image = load_image(image_path).resize((1280, 720))
    w, h = image.size
    processor = CannyDetector()
    control_image = processor(
        image, low_threshold=50, high_threshold=200, detect_resolution=1280, image_resolution=1280
    )
    prompt = "interior design, a living room in Japanese home style, realistic, ambient lighting"

    image = pipe(
        prompt,
        control_image=control_image,
        controlnet_conditioning_scale=0.7,
        num_inference_steps=40,
        guidance_scale=3.5,
        height=h,
        width=w,
        generator=generator,
        num_images_per_prompt=1,
    ).images[0]

    image.save(image_path.replace(working_dir, "outputs/ConCanny"))
    control_image.save(image_path.replace(working_dir, "outputs/ConCanny/control_images"))
