import argparse
import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxControlPipeline, FluxControlImg2ImgPipeline, FluxPriorReduxPipeline
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor
from controlnet_aux import CannyDetector
import os
from pathlib import Path
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Unified FLUX Pipeline")
    parser.add_argument("pipeline", type=str, 
                        choices=["text2img", "img2img", "canny", "depth", "redux", "depth_img2img",
                                 "canny_img2img", "depth_redux", "canny_redux", "canny_depth", "canny_depth_redux"],
                        help="Choose the FLUX pipeline to use")
    parser.add_argument("--input_image", type=str, help="Path to the input image", default="inputs/a5.png")
    parser.add_argument("--prompt", type=str, help="Text prompt for image generation", default="interior design, bedroom, ambient lighting")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the output image")
    return parser.parse_args()

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    fx = max(w/1920, h/1080)
    w, h = int(w/fx), int(h/fx)
    image = image.resize((w, h))
    return image, w, h

def load_pipeline(pipeline_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    if pipeline_name in ["text2img", "redux"]:
        pipe = FluxPipeline.from_pretrained("./models/FLUX.1-dev", torch_dtype=dtype).to(device)
    elif pipeline_name in ["img2img"]:
        pipe = FluxImg2ImgPipeline.from_pretrained("./models/FLUX.1-dev", torch_dtype=dtype).to(device)
    elif pipeline_name in ["canny", "canny_redux"]:
        pipe = FluxControlPipeline.from_pretrained("./models/FLUX.1-Canny-dev", torch_dtype=dtype).to(device)
    elif pipeline_name in ["depth", "depth_redux"]:
        pipe = FluxControlPipeline.from_pretrained("./models/FLUX.1-Depth-dev", torch_dtype=dtype).to(device)
    elif pipeline_name in ["canny_img2img"]:
        pipe = FluxControlImg2ImgPipeline.from_pretrained("./models/FLUX.1-Canny-dev", torch_dtype=dtype).to(device)
    elif pipeline_name in ["depth", "depth_img2img"]:
        pipe = FluxControlImg2ImgPipeline.from_pretrained("./models/FLUX.1-Depth-dev", torch_dtype=dtype).to(device)
    else:
        raise NotImplementedError(f"Unsupported pipeline: {pipeline_name}")

    if pipeline_name in ["redux", "depth_redux", "canny_redux"]:
        pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained("./models/FLUX.1-Redux-dev", torch_dtype=dtype).to(device)
        return pipe, pipe_prior_redux
    else:
        return pipe, None

def main():
    args = parse_args()

    # Process input image
    image, width, height = process_image(args.input_image)

    # Load the appropriate pipeline
    pipe, pipe_prior_redux = load_pipeline(args.pipeline)

    # Prepare control image if needed
    control_image = None
    if args.pipeline in ["canny", "canny_img2img", "canny_redux", "depth", "depth_img2img", "depth_redux"]:
        control_image = image.copy()
        if "depth" in args.pipeline:
            processor = DepthPreprocessor.from_pretrained("./models/depth-anything-large-hf")
            control_image = processor(control_image)[0].convert("RGB")
        if "canny" in args.pipeline:
            processor = CannyDetector()
            control_image = processor(control_image, low_threshold=100, high_threshold=200)

    inputs = {}
    if pipe_prior_redux is not None:
        pipe_prior_output = pipe_prior_redux(image, prompt=args.prompt)
        inputs = {**inputs, **pipe_prior_output}
    else:
        inputs["prompt"] = args.prompt
    if control_image is not None:
        inputs["control_image"] = control_image
    if "img2img" in args.pipeline:
        inputs["image"] = image
        inputs["strength"] = 0.8

    # Generate image
    output = pipe(
        width=width,
        height=height,
        guidance_scale=7,
        num_inference_steps=50,
        **inputs
        )

    # Save the output image
    output_dir = os.path.join(args.output_dir, args.pipeline)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{Path(args.input_image).stem}.png"
    output_path = os.path.join(output_dir, output_filename)
    output.images[0].save(output_path)
    print(f"Output image saved to: {output_path}")

if __name__ == "__main__":
    main()