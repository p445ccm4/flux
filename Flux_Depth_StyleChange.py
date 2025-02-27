import json
import torch
from diffusers.utils import load_image
from diffusers import FluxControlPipeline
from image_gen_aux import DepthPreprocessor
import os
from PIL import Image

generator = torch.Generator(device="cuda").manual_seed(87544357)

processor = DepthPreprocessor.from_pretrained("./models/depth-anything-large-hf")

pipe = FluxControlPipeline.from_pretrained("./models/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.to("cuda")

working_dir = "inputs"
output_dir_control = "outputs/Depth/control_images"
output_dir_gen = "outputs/Depth"

os.makedirs(output_dir_control, exist_ok=True)
os.makedirs(output_dir_gen, exist_ok=True)

# Define input size variables
SIMPLE_RESIZE_WIDTH = 1280
SIMPLE_RESIZE_HEIGHT = 720


def resize_and_pad_image(image, resize_width, resize_height): # Added target_size argument with default value
    """Resizes an image to fit within target_size x target_size while maintaining aspect ratio and pads it to target_size x target_size.

    Args:
        image (PIL.Image.Image): Input image.
        target_size (int): The target size for padding and resizing. Defaults to PAD_RESIZE_SIZE.

    Returns:
        tuple[PIL.Image.Image, int, int]: Padded image, offset_x, offset_y.
    """
    original_w, original_h = image.size
    aspect_ratio = original_w / original_h

    if original_w > original_h:
        new_w = resize_width
        new_h = int(resize_width / aspect_ratio)
    else:
        new_h = resize_height
        new_w = int(resize_height * aspect_ratio)

    resized_image = image.resize((new_w, new_h))

    padded_image = Image.new("RGB", (resize_width, resize_height), (255, 255, 255))  # White background
    offset_x = (resize_height - new_w) // 2
    offset_y = (resize_height - new_h) // 2
    padded_image.paste(resized_image, (offset_x, offset_y))
    return padded_image, offset_x, offset_y, new_w, new_h


with open("inputs/style_prompts.json", "r") as f:
        data = json.load(f)
        styles_data = data["styles"]

for image_path in sorted([os.path.join(working_dir, f) for f in os.listdir(working_dir) if f.endswith((".png", ".jpg", ".jpeg"))]):
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        original_image = load_image(image_path)
        original_w, original_h = original_image.size

        # Use the resize and pad function
        # image, offset_x, offset_y, new_w, new_h = resize_and_pad_image(original_image, SIMPLE_RESIZE_WIDTH, SIMPLE_RESIZE_HEIGHT) # <--- Function call using variable
        image = original_image.resize((SIMPLE_RESIZE_WIDTH, SIMPLE_RESIZE_HEIGHT)) # Simple resize using variables

        control_image = processor(image)[0].convert("RGB")
        control_image.save(os.path.join(output_dir_control, f"{image_basename}_control.png"))

        for style_info in styles_data:
            style_name = style_info["style_name"]
            prompt = "very realistic, very photorealistic, " + style_info["prompt"]

            generated_image = pipe(
                prompt,
                control_image=control_image,
                num_inference_steps=50,
                guidance_scale=7,
                height=SIMPLE_RESIZE_HEIGHT,
                width=SIMPLE_RESIZE_WIDTH,
                generator=generator,
            ).images[0]

            # generated_image = generated_image.crop((offset_x, offset_y, offset_x + new_w, offset_y + new_h))

            final_image = generated_image.resize((original_w, original_h), Image.LANCZOS)
            final_image.save(os.path.join(output_dir_gen, f"{image_basename}_{style_name}.png"))

print("Processing complete. Images saved in outputs/ConDepth.")