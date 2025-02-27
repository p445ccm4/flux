import json
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers import FluxControlNetPipeline
from controlnet_aux import CannyDetector
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
with open("inputs/style_prompts.json", "r") as f:
        data = json.load(f)
        styles_data = data["styles"]

for image_path in sorted([os.path.join(working_dir, f) for f in os.listdir(working_dir) if f.endswith((".png", ".jpg", ".jpeg"))]):
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        image = load_image(image_path).resize((1280, 720))
        w, h = image.size
        processor = CannyDetector()
        control_image = processor(
                image, low_threshold=100, high_threshold=250, detect_resolution=1024, image_resolution=1024
        )
        control_image.save(image_path.replace(working_dir, "outputs/ConCanny/control_images"))

        for style_info in styles_data:
            style_name = style_info["style_name"]
            prompt = "realistic, " + style_info["prompt"]

            image = pipe(
                prompt,
                control_image=control_image,
                controlnet_conditioning_scale=0.9,
                num_inference_steps=40,
                guidance_scale=3.5,
                height=h,
                width=w,
                generator=generator,
                num_images_per_prompt=1,
            ).images[0]

            image.save(os.path.join("outputs/ConCanny", f"{image_basename}_{style_name}.png"))
