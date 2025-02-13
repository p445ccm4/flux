import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
import numpy as np
from PIL import Image

image_path = "./inputs/b2.jpeg"
image = load_image(image_path).resize((1920, 1080))
# mask = load_image("inputs/b0_mask.png").resize(image.size)
w, h = image.size

pipe = FluxFillPipeline.from_pretrained("./models/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")

# Lower half
mask_array = np.zeros((h, w), dtype=np.uint8)
mask_array[h//2:, :] = 255
mask = Image.fromarray(mask_array)
image = pipe(
    prompt="a living room filled with furniture. photorealistic modern living room, featuring a sleek grey sectional sofa, a glass and chrome coffee table, and a minimalist fireplace. Neutral color palette of beige, white, and grey with accents of muted teal. Large windows letting in natural light, complemented by a modern arc floor lamp. Potted fiddle leaf fig plant, abstract art on the wall, textured cream rug.",
    image=image,
    mask_image=mask,
    height=h,
    width=w,
    guidance_scale=70,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(image_path.replace(".jpeg", "_flux_lower.jpeg"))

# Upper half
mask_array = np.zeros((h, w), dtype=np.uint8)
mask_array[:h//2, :] = 255
mask = Image.fromarray(mask_array)
image = pipe(
    prompt="a living room filled with furniture. photorealistic modern living room, featuring a sleek grey sectional sofa, a glass and chrome coffee table, and a minimalist fireplace. Neutral color palette of beige, white, and grey with accents of muted teal. Large windows letting in natural light, complemented by a modern arc floor lamp. Potted fiddle leaf fig plant, abstract art on the wall, textured cream rug.",
    image=image,
    mask_image=mask,
    height=h,
    width=w,
    guidance_scale=70,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(image_path.replace(".jpeg", "_flux_upper.jpeg"))
