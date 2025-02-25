import torch
from FLUXControlnetInpainting import FluxControlNetModel, FluxTransformer2DModel, FluxControlNetInpaintingPipeline
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import cv2
import os

# Define the directory containing the images
image_dir = "./inputs/1key_clear"
image_list = sorted(os.listdir(image_dir))

# Delete all files with undesired suffixes
suffices_to_delete = ["_1key_clear.png", "_binary_mask.png"]
for filename in image_list:
    if any(filename.endswith(suffix) for suffix in suffices_to_delete):
        os.remove(os.path.join(image_dir, filename))

# Define the desired and undesired suffixes
desired_suffix = ".png"
undesired_suffixes = ["masked_image.jpg", "1key_clear.png", "binary_mask.png"]
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

# Loop through all desired files in the directory
for filename in image_list:
    if filename.endswith(desired_suffix) and not any(
        filename.endswith(suffix) for suffix in undesired_suffixes
    ):
        image_path = os.path.join(image_dir, filename)
        print(f"Processing {image_path}...")
        image = load_image(image_path).resize((1280, 720))
        w, h = image.size

        # Convert the mask to grayscale
        mask = load_image(image_path.replace(desired_suffix, "_masked_image.jpg")).convert("L")

        # Threshold the mask
        mask_array = np.array(mask)
        print(mask_array.shape)
        threshold = 255
        binary_mask = np.where((mask_array < threshold), 0, 1)
        # Save the binary mask
        mask = Image.fromarray(np.uint8(binary_mask * 255))
        # Convert the binary mask to an OpenCV image
        cv_mask = np.array(mask)

        # Define the kernel for erosion
        kernel = np.ones((5, 5), np.uint8)

        # Perform dilation
        dilation = cv2.dilate(cv_mask, kernel, iterations=5)

        # Convert the eroded mask back to a PIL image
        mask = Image.fromarray(dilation)
        mask.save(image_path.replace(desired_suffix, "_binary_mask.png"))

        prompt="an empty room, blank room, clean, nothing, clear, plain, bright, light"

        # Run the pipeline
        flux_image = pipe(
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
        flux_image.save(image_path.replace(desired_suffix, "_1key_clear_inpaint.png"))
