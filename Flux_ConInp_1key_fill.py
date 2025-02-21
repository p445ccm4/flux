import torch
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import os
from FLUXControlnetInpainting import FluxControlNetModel, FluxTransformer2DModel, FluxControlNetInpaintingPipeline

# Directory containing input images
image_dir = "inputs/1key_fill"

# Get list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png")) and not "flux" in f]

# Build pipeline
controlnet = FluxControlNetModel.from_pretrained("./models/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
        "./models/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
    )
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "./models/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)

prompts = [
    "ceiling fan, lights, pictures",
    "sofa, chair, table"
]

for image_file in sorted(image_files):
    # Load image
    image_path = os.path.join(image_dir, image_file)
    image = load_image(image_path).resize((1280, 720))
    w, h = image.size

    for i in [1, 0]:
        for j in [1, 0]:
            # Create mask for the current quadrant
            mask_array = np.zeros((h, w), dtype=np.uint8)
            mask_array[h//2*i:h//2*(i+1), w//2*j:w//2*(j+1)] = 255
            mask = Image.fromarray(mask_array)

            # image = np.array(image)
            # image[h//2*i:h//2*(i+1), w//2*j:w//2*(j+1)] = 255
            # image = Image.fromarray(image)

            # Define the output filename for the current quadrant
            output_filename = image_path.replace(os.path.splitext(image_file)[1], f"_flux_coninp_q{i*2+j+1}.jpeg")

            # Modify the prompt based on the row index 'i'
            if i == 0:
                prompt = f"a {prompts[0]} on the ceiling of a living room. photorealistic modern living room, featuring {prompts[0]}. Neutral color palette of beige, white, and grey with accents of muted teal. Large windows letting in natural light, complemented by a modern arc floor lamp. Potted fiddle leaf fig plant, abstract art on the wall, textured cream rug."
            else:
                prompt = f"a {prompts[1]} on the floor of a living room. photorealistic modern living room, featuring {prompts[1]}. Neutral color palette of beige, white, and grey with accents of muted teal. Large windows letting in natural light, complemented by a modern arc floor lamp. Potted fiddle leaf fig plant, abstract art on the wall, textured cream rug."

            # Run the pipeline
            image = pipe(
                prompt=prompt,
                control_image=image,
                control_mask=mask,
                height=h,
                width=w,
                num_inference_steps=28,
                generator=torch.Generator(device="cuda").manual_seed(24),
                controlnet_conditioning_scale=0.9,
                guidance_scale=3.5,
                negative_prompt="",
                true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
            ).images[0]

            # Save the image
            image.save(output_filename)
