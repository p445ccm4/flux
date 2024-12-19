import torch
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image

input_image = "inputs/b0.png"

device = "cuda"
pipe = FluxImg2ImgPipeline.from_pretrained("./models/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

init_image = load_image(input_image)
w, h = init_image.size

prompt = "many furniture, living room, interior design, clean, tidy, ambient lighting"

image = pipe(
    prompt=prompt,
    image=init_image, 
    num_inference_steps=50, 
    strength=0.7, 
    guidance_scale=30.0,
    width=w, 
    height=h,
).images[0]
image.save("outputs/b0.png")