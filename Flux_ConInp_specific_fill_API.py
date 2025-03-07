import torch
import numpy as np
from PIL import Image
from FLUXControlnetInpainting import FluxControlNetModel, FluxTransformer2DModel, FluxControlNetInpaintingPipeline
from openai import OpenAI
import time

def specific_fill(image_np, room_type="living room", specific_requirement=""):
    # Get the prompt
    start = time.time()
    deepseek = DeepSeek()
    prompt = deepseek.gen_prompt(room_type + ". " + specific_requirement)
    print(f"Prompt suggested by DeepSeek: {prompt}")
    print(f"Time taken to generate prompt: {time.time()-start}s")

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

    image = Image.fromarray(image_np)
    w, h = image.size
    margin = 0

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

        # Run the pipeline
        image = pipe(
            prompt=prompt,
            control_image=image,
            control_mask=mask,
            height=h//8*8,
            width=w//8*8,
            num_inference_steps=40,
            generator=torch.Generator(device="cuda").manual_seed(24),
            controlnet_conditioning_scale=0.9,
            guidance_scale=3.5,
            negative_prompt="blurry, low quality, human, logo",
            true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
        ).images[0]

    image_np = np.array(image)
    return image_np


class DeepSeek():
    def __init__(self):
        self.client = OpenAI(
            api_key="sk-f14f2d1f72f54c348f3fd325ca0e2ba0",
            base_url="https://api.deepseek.com"
            )
        print("DeepSeek API initialized.")

    def gen_prompt(self, input: str):
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", 
                 "content": """
                 You are making prompt to a text-to-image generation model to generate 
                 interior design images. You must Reply with a string of English prompt 
                 with no quotation marks within 70 words. You must specify the room type and
                 what furniture should be added to the room, both on the ceiling and on the floor. 
                 On top of the user's input, include the furniture that should be added to the room
                 in order to make the place resonably sufficient to be lived/work in.
                 For example: 'A kitchen. Pendant Light, Recessed Lights, Range Hood, Upper Cabinets, 
                 Window Treatments on the ceiling. Base Cabinets, Kitchen Island, Refrigerator 
                 and Oven Sink on the floor.'
                 """},
                {"role": "user", "content": input},
            ],
            stream=False
        )

        return response.choices[0].message.content

if __name__ == "__main__":
    image_path = "inputs/specific_fill/b1.jpeg"
    image_np = np.array(Image.open(image_path).convert("RGB").resize((1280, 720)))

    specific_requirement = "木製浴缸，洗手盆有镜子和燈"

    with open("inputs/specific_fill/b1.txt", "w") as f:
        f.write(specific_requirement)

    filled_image_np = specific_fill(
        image_np, 
        room_type="toilet/bathroom", 
        specific_requirement=specific_requirement
        )

    Image.fromarray(filled_image_np).save("inputs/specific_fill/b1_filled.png")