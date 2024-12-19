import torch
from diffusers import FluxPriorReduxPipeline, FluxPipeline, FluxControlPipeline
from diffusers.utils import load_image
import argparse
import os
from pathlib import Path
from controlnet_aux import CannyDetector

# Add command line argument parsing
parser = argparse.ArgumentParser(description="FLUX Redux image generation")
parser.add_argument("--input_image", type=str, help="Path to the input image", default="inputs/a5.png")
parser.add_argument("--output_dir", type=str, help="Directory to save the output image", default="outputs/Flux_Redux_dev/")
args = parser.parse_args()
device = "cuda"
dtype = torch.bfloat16

repo_redux = "./models/FLUX.1-Redux-dev"
repo_base = "./models/FLUX.1-Canny-dev"
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(repo_redux, torch_dtype=dtype).to(device)
pipe = FluxControlPipeline.from_pretrained(
    repo_base, 
    text_encoder=None, 
    text_encoder_2=None, 
    torch_dtype=torch.bfloat16
).to(device)

# Load the input image from the provided path
image = load_image(args.input_image)
pipe_prior_output = pipe_prior_redux(image)
w, h = image.size
fx = min(w/1920, h/1080)
w, h = int(w/fx), int(h/fx)
image = image.resize((w, h))

processor = CannyDetector()
control_image = processor(
    image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024
)

images = pipe(
    control_image=control_image,
    guidance_scale=7,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0),
    height=h,
    width=w,
    **pipe_prior_output,
).images

# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Generate the output filename
input_filename = Path(args.input_image).stem
output_filename = f"{input_filename}.png"
output_path = os.path.join(args.output_dir, output_filename)

# Save the output image
images[0].save(output_path)
print(f"Output image saved to: {output_path}")