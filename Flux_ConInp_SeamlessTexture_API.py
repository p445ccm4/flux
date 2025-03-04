import cv2
import torch
from FLUXControlnetInpainting import FluxControlNetModel, FluxTransformer2DModel, FluxControlNetInpaintingPipeline
from diffusers.utils import load_image
from PIL import ImageOps, Image
import numpy as np

def pad_image(image, w_border, h_border):
    padded_image = ImageOps.expand(image, border=(w_border, h_border), fill="black")
    return padded_image

def rearrange_quadrants(image_np):
    h_mid, w_mid = image_np.shape[0] // 2, image_np.shape[1] // 2
    top_left = image_np[:h_mid, :w_mid]
    top_right = image_np[:h_mid, w_mid:]
    bottom_left = image_np[h_mid:, :w_mid]
    bottom_right = image_np[h_mid:, w_mid:]

    rearranged_image_np = np.zeros_like(image_np)
    rearranged_image_np[:h_mid, :w_mid] = bottom_right
    rearranged_image_np[:h_mid, w_mid:] = bottom_left
    rearranged_image_np[h_mid:, :w_mid] = top_right
    rearranged_image_np[h_mid:, w_mid:] = top_left

    return rearranged_image_np

def create_repeated_image(image_np, h_border, w_border):
    h, w, c = image_np.shape
    repeated_image_np = np.ones((h * 2 + h_border*2, w * 2 + w_border*2, c), dtype=image_np.dtype)*255

    repeated_image_np[:h, :w] = image_np
    repeated_image_np[:h, w + w_border*2:] = image_np
    repeated_image_np[h + h_border*2:, :w] = image_np
    repeated_image_np[h + h_border*2:, w + w_border*2:] = image_np

    return repeated_image_np

def create_mask(h, w, h_border, w_border):
    mask = np.zeros((h * 2 + h_border*2, w * 2 + w_border*2), dtype=np.uint8)
    mask[h:h + h_border*2, :] = 255
    mask[:, w:w + w_border*2] = 255

    mask_image = Image.fromarray(mask)
    return mask_image

def create_prompt(image):
    from transformers import AutoProcessor, AutoModelForCausalLM
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained("./models/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("./models/Florence-2-large", trust_remote_code=True)

    prompt = "<CAPTION>"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=4096,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task="<CAPTION>", image_size=(image.width, image.height))
    description = "A seamless texture following the oringinal pattern. It is continuous with no borders and breakings. " + parsed_answer[prompt]
    description = parsed_answer[prompt] + " A seamless texture following the oringinal pattern. The pattern is continuous with no visible borders, breakings and boundaries."

    return description

def flux_fill(image, mask, prompt, h, w):
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

    flux_image = pipe(
        prompt=prompt,
        control_image=image,
        control_mask=mask,
        height=h//8*8,
        width=w//8*8,
        num_inference_steps=28,
        generator=torch.Generator(device="cuda").manual_seed(24),
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="borders, breaking, boundaries, discontinuity",
        true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
    ).images[0]
    return flux_image

def create_seamless_check(image_np, repeat_count=3):
    """
    Creates a seamless check pattern from a NumPy image array using OpenCV and NumPy.

    Args:
        image_np (numpy.ndarray): Input image as a NumPy array.
        repeat_count (int): Number of times to repeat the image horizontally and vertically.

    Returns:
        numpy.ndarray: Repeated image as a NumPy array.
    """
    resized_image_np = cv2.resize(image_np, (300, 300))
    repeated_image_np = np.zeros((300 * repeat_count, 300 * repeat_count, 3), dtype=np.uint8)

    for i in range(repeat_count):
        for j in range(repeat_count):
            x_offset = i * 300
            y_offset = j * 300
            repeated_image_np[y_offset:y_offset+300, x_offset:x_offset+300] = resized_image_np

    return repeated_image_np

def process_seamless_texture(image_np: np.ndarray):
    h, w = image_np.shape[:2]
    h_border, w_border = h // 10, w // 10 # portion of padding

    repeated_image_np = create_repeated_image(image_np, h_border, w_border)
    
    # Convert to PIL
    repeated_image_PIL = Image.fromarray(repeated_image_np)
    mask = create_mask(h, w, h_border, w_border)
    prompt = create_prompt(repeated_image_PIL)
    flux_image = flux_fill(repeated_image_PIL, mask, prompt, (h+h_border)*2, (w+w_border)*2)
    cropped_image = flux_image.crop((w//2, h//2, w//2*3+w_border*2, h//2*3+h_border*2))

    # Convert to numpy array
    cropped_image_np = np.array(cropped_image)
    final_image_np = rearrange_quadrants(cropped_image_np)

    return final_image_np

if __name__ == "__main__":
    image_np = cv2.imread("inputs/Texture/a1.jpg")
    image_np = cv2.resize(image_np, (800, 800))

    result_np_image = process_seamless_texture(image_np)
    cv2.imwrite("outputs/a1_seamless.jpg", result_np_image)

    seamless_check_image = create_seamless_check(result_np_image)
    cv2.imwrite("outputs/a1_seamless_check.jpg", seamless_check_image)