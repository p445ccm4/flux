import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import ImageOps, Image
import numpy as np
import cv2

image_path = "../ViewCrafter/outputs/panorama/d5.png"
image = load_image(image_path)
w, h = image.size

if w > h*2:
    h = w//2
else:
    w = h*2

w, h = int(w//2), int(h//2)

padded_image = ImageOps.pad(image, (w, h), color="black")
padded_image.save(image_path.replace(".png", "_padded_image.png"))

padded_mask = cv2.inRange(np.array(padded_image), np.array([0, 0, 0]), np.array([0, 0, 0]))
padded_mask = cv2.dilate(padded_mask, np.ones((5, 5), np.uint8))
# _, padded_mask = cv2.threshold(padded_mask, 1, 255, cv2.THRESH_BINARY)
cv2.imwrite(image_path.replace(".png", "_padded_mask.png"), padded_mask)

pipe = FluxFillPipeline.from_pretrained("./models/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")
image = pipe(
    prompt="panorama image of an interior design image",
    image=padded_image,
    mask_image=padded_mask,
    height=h,
    width=w,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(image_path.replace(".png", "_flux.png"))
