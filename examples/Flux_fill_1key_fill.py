import torch
from diffusers import FluxInpaintPipeline
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import os

# Directory containing input images
image_dir = "inputs/1key_fill"

# Get list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png")) and not "flux" in f]

pipe = FluxInpaintPipeline.from_pretrained("./models/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")

prompts = [
    "ceiling fan, lights, pictures",
    "sofa, chair, table"
]

for image_file in sorted(image_files):
    # Load image
    image_path = os.path.join(image_dir, image_file)
    image = load_image(image_path).resize((1920, 1080))
    w, h = image.size

    for i in [1, 0]:
        for j in [1, 0]:
            # Create mask for the current quadrant
            mask_array = np.zeros((h, w), dtype=np.uint8)
            mask_array[h//2*i:h//2*(i+1), w//2*j:w//2*(j+1)] = 255
            mask = Image.fromarray(mask_array)

            # Define the output filename for the current quadrant
            output_filename = image_path.replace(os.path.splitext(image_file)[1], f"_flux_dev_q{i*2+j+1}.jpeg")

            # Modify the prompt based on the row index 'i'
            if i == 0:
                prompt = f"a {prompts[0]} on the ceiling of a living room. photorealistic modern living room, featuring {prompts[0]}. Neutral color palette of beige, white, and grey with accents of muted teal. Large windows letting in natural light, complemented by a modern arc floor lamp. Potted fiddle leaf fig plant, abstract art on the wall, textured cream rug."
            else:
                prompt = f"a {prompts[1]} on the floor of a living room. photorealistic modern living room, featuring {prompts[1]}. Neutral color palette of beige, white, and grey with accents of muted teal. Large windows letting in natural light, complemented by a modern arc floor lamp. Potted fiddle leaf fig plant, abstract art on the wall, textured cream rug."

            # Run the pipeline
            image = pipe(
                prompt=prompt,
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
