import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import ImageOps, Image

image = load_image("./inputs/meme material/loming.jpg")
w, h = image.size
image = image.resize((w//2, h//2))
padded_image = ImageOps.expand(image, border=400, fill='white')
mask = Image.new("L", image.size, 0)
padded_mask = ImageOps.expand(mask, border=400, fill='white')
w, h = padded_image.size
padded_image.save("padded_image.jpg")
padded_mask.save("padded_mask.jpg")

pipe = FluxFillPipeline.from_pretrained("./models/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")
image = pipe(
    prompt="",
    image=padded_image,
    mask_image=padded_mask,
    height=h,
    width=w,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"flux-fill-dev.png")
