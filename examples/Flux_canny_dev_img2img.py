import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlImg2ImgPipeline
from diffusers.utils import load_image

pipe = FluxControlImg2ImgPipeline.from_pretrained(
    "./models/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "full of furniture, living room, interior design, clean, tidy, ambient lighting"
image = load_image(
    "inputs/b0.png"
)
control_image = load_image(
    "inputs/b0.png"
)

processor = CannyDetector()
control_image = processor(
    control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024
)

image = pipe(
    prompt=prompt,
    image=image,
    control_image=control_image,
    strength=0.8,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=30.0,
).images[0]
image.save("outputs/b0.png")