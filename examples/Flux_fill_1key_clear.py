import torch
from diffusers import FluxInpaintPipeline
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
pipe = FluxInpaintPipeline.from_pretrained(
"./models/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

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

        flux_image = pipe(
            prompt="an empty room, blank room, clean, nothing, clear, plain, bright, light",
            image=image,
            mask_image=mask,
            height=h,
            width=w,
            guidance_scale=70,
            num_inference_steps=50,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        # Save the image
        flux_image.save(image_path.replace(desired_suffix, "_1key_clear_inpaint.png"))
