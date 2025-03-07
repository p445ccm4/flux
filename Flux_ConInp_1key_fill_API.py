import torch
import numpy as np
from PIL import Image
from FLUXControlnetInpainting import FluxControlNetModel, FluxTransformer2DModel, FluxControlNetInpaintingPipeline

room_prompts = {
    "dining room":[
        "Pendant Light, Recessed Lights, Wall Sconces, Window Treatments, Art",
        "Dining Table, Chairs, Rug, Sideboard"
    ],
    "toilet":[
        "Ceiling Light, Ventilation Fan, Mirror, Shower Head, Window Treatments, Towel Rack",
        "Toilet, Vanity, Bathtub, Shower, Sink, Floor, Toilet Paper Holder, Waste Bin"
    ],
    "living room":[
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

def one_key_fill(image_np, room_type="living room"):
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

    image = Image.fromarray(image_np)
    w, h = image.size
    margin = 0

    for t in range(2):
        # Get the prompt
        prompts = room_prompts[room_type]
        prompt = f" A {room_type}, {prompts[0]} on the ceiling. {prompts[1]} on the floor."

        # Create mask for the current quadrant
        mask_array = np.zeros((h, w), dtype=np.uint8)
        if t == 0:
            mask_array[h//2*0:h//2*(0+1)+margin, w//2*0:w//2*(0+1)+margin] = 255
            mask_array[h//2*1-margin:h//2*(1+1), w//2*1-margin:w//2*(1+1)] = 255
        else:
            mask_array[h//2*0:h//2*(0+1)+margin, w//2*1-margin:w//2*(1+1)] = 255
            mask_array[h//2*1-margin:h//2*(1+1), w//2*0:w//2*(0+1)+margin] = 255
        mask = Image.fromarray(mask_array)

        # Run the pipeline
        image = pipe(
            prompt=prompt,
            control_image=image,
            control_mask=mask,
            height=h//8*8,
            width=w//8*8,
            num_inference_steps=28,
            generator=torch.Generator(device="cuda").manual_seed(24),
            controlnet_conditioning_scale=0.9,
            guidance_scale=3.5,
            negative_prompt="blurry, low quality, human, logo",
            true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
        ).images[0]

    image_np = np.array(image)
    return image_np

if __name__ == "__main__":
    image_path = "inputs/1key_fill/b3.jpg"
    image_np = np.array(Image.open(image_path).convert("RGB").resize((1280, 720)))

    filled_image_np = one_key_fill(image_np, room_type="living room")

    Image.fromarray(filled_image_np).save("filled_image.jpg")