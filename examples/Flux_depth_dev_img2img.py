import torch
from image_gen_aux import DepthPreprocessor
from diffusers import FluxControlImg2ImgPipeline
from diffusers.utils import load_image

pipe = FluxControlImg2ImgPipeline.from_pretrained(
    "./models/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "full of furniture, living room, interior design, clean, tidy, ambient lighting, realistic, natural, high-quality"
image = load_image(
    "inputs/b0.png"
)
control_image = load_image(
    "inputs/b0.png"
)

processor = DepthPreprocessor.from_pretrained("./models/depth-anything-large-hf")
control_image = processor(control_image)[0].convert("RGB")

image = pipe(
    prompt=prompt,
    image=image,
    control_image=control_image,
    strength=0.9,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=30.0,
).images[0]
image.save("outputs/b0.png")