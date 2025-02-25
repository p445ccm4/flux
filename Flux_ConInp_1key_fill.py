import torch
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import os
from FLUXControlnetInpainting import FluxControlNetModel, FluxTransformer2DModel, FluxControlNetInpaintingPipeline

# Directory containing input images
image_dir = "inputs/1key_fill"

# Get list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png")) and not "flux" in f]

# Build pipeline
controlnet = FluxControlNetModel.from_pretrained("./models/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
        "./models/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
    )
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "./models/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)

room_prompts = {
    "dining_room":[
        "Pendant Light, Recessed Lights, Wall Sconces, Window Treatments, Art",
        "Dining Table, Chairs, Rug, Sideboard"
    ],
    "toilet":[
        "Ceiling Light, Ventilation Fan, Mirror, Shower Head, Window Treatments, Towel Rack",
        "Toilet, Vanity, Bathtub, Shower, Sink, Floor, Toilet Paper Holder, Waste Bin"
    ],
    "living_room":[
        "Ceiling Light, Recessed Lighting, Wall Sconces, Window Treatments, Wall Art",
        "Sofa, Chairs, Coffee Table, Rug, TV Stand, Floor Lamps, Side Table"
    ],
    "bedroom":[
        "Ceiling Light, Recessed Lighting, Wall Sconces, Window Treatments, Mirror, Wall Art",
        "Bed, Nightstands, Rug, Dresser, Floor Lamps, Bench"
    ],
    "kitchen":[
        "Pendant Light, Recessed Lights, Range Hood, Upper Cabinets, Window Treatments",
        "Base Cabinets, Kitchen Island, Refrigerator, Oven, Sink, Floor"
    ]
}

for image_file in sorted(image_files):
    # Load image
    image_path = os.path.join(image_dir, image_file)
    image = load_image(image_path)
    org_w, org_h = image.size
    empty_image = image.resize((1280, 720))
    w, h = empty_image.size
    margin = 0
    n = 8

    for room_type, prompts in room_prompts.items():
        images = [empty_image.copy() for _ in range(n)]
        for t in range(2):
            # Create mask for the current quadrant
            mask_array = np.zeros((h, w), dtype=np.uint8)
            if t == 0:
                mask_array[h//2*0:h//2*(0+1)+margin, w//2*0:w//2*(0+1)+margin] = 255
                mask_array[h//2*1-margin:h//2*(1+1), w//2*1-margin:w//2*(1+1)] = 255
            else:
                mask_array[h//2*0:h//2*(0+1)+margin, w//2*1-margin:w//2*(1+1)] = 255
                mask_array[h//2*1-margin:h//2*(1+1), w//2*0:w//2*(0+1)+margin] = 255
            mask = Image.fromarray(mask_array)

            # Modify the prompt based on the row index 'i'
            prompt = f" A {room_type}, {prompts[0]} on the ceiling. {prompts[1]} on the floor."

            # Run the pipeline
            images = pipe(
                prompt=[prompt] * n,
                control_image=[image] * n,
                control_mask=[mask] * n,
                height=h,
                width=w,
                num_inference_steps=28,
                generator=torch.Generator(device="cuda").manual_seed(24),
                controlnet_conditioning_scale=0.9,
                guidance_scale=3.5,
                negative_prompt="",
                true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
            ).images

        # Save the image
        for i in range(n):
            # Define the output filename for the current quadrant
            output_filename = image_path.replace(os.path.splitext(image_file)[1], f"_flux_coninp_{room_type.replace(" ", "_")}_{i}.jpeg")
            images[i].resize((org_w, org_h)).save(output_filename)
