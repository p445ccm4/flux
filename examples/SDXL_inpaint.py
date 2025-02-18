import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
import numpy as np
from PIL import Image

image_path = "./inputs/b2.jpeg"
image = load_image(image_path).resize((1920, 1080))
w, h = image.size

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "./models/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

# Split into quadrants
for i in range(2):
    for j in range(2):
        mask_array = np.zeros((h, w), dtype=np.uint8)
        mask_array[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2] = 255
        mask = Image.fromarray(mask_array)
        image = pipe(
            prompt="a living room filled with furniture. photorealistic modern living room, featuring a sleek grey sectional sofa, a glass and chrome coffee table, and a minimalist fireplace. Neutral color palette of beige, white, and grey with accents of muted teal. Large windows letting in natural light, complemented by a modern arc floor lamp. Potted fiddle leaf fig plant, abstract art on the wall, textured cream rug.",
            image=image,
            mask_image=mask,
            num_inference_steps=50,
            strength=0.80,
        ).images[0]
        image.save(image_path.replace(".jpeg", f"_sdxl_q{i}{j}.jpeg"))
