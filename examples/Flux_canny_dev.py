import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
import cv2
import numpy as np

pipe = FluxControlPipeline.from_pretrained("./models/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16).to("cuda")

prompt = "full of furniture, living room, interior design, clean, tidy, ambient lighting, realistic, natural, high-quality"
control_image = load_image("./inputs/a1.jpeg")
w, h = control_image.size

# processor = CannyDetector()
# control_image = processor(control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024)
# control_image.save("./outputs/a0_canny.jpg")
control_image = np.array(control_image)
control_image = cv2.resize(control_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
control_image = cv2.Canny(control_image, 50, 200)
control_image = np.stack([control_image, control_image, control_image], axis=2)
cv2.imwrite("./outputs/a1_canny.jpg", control_image)

image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=h,
    width=w,
    num_inference_steps=50,
    guidance_scale=30.0,
).images[0]
image.save("outputs/a1.png")
