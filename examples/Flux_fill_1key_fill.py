import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
import numpy as np
from PIL import Image

image_path = "./inputs/b1.jpeg"
image = load_image(image_path).resize((1280, 720))
# mask = load_image("inputs/b0_mask.png").resize(image.size)
w, h = image.size

pipe = FluxFillPipeline.from_pretrained("./models/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")

for i in [1, 0]:
    for j in [1, 0]:
        # Create mask for the current quadrant
        mask_array = np.zeros((h, w), dtype=np.uint8)
        mask_array[h//2*i:h//2*(i+1), w//2*j:w//2*(j+1)] = 255
        mask = Image.fromarray(mask_array)

        # Define the output filename for the current quadrant
        output_filename = image_path.replace(".jpeg", f"_flux_q{i*2+j+1}.jpeg")

        # Run the pipeline
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

        # Save the image
        image.save(output_filename)
