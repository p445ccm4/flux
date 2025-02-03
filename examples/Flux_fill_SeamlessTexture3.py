import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import ImageOps, Image
import numpy as np
import os
import argparse

def pad_image(image, w_border, h_border, output_path):
    padded_image = ImageOps.expand(image, border=(w_border, h_border), fill="black")
    padded_output_path = output_path + "_1_padded_image.png"
    padded_image.save(padded_output_path)
    print(f"Padded image saved at: {padded_output_path}")
    return padded_image

def rearrange_quadrants(image, output_path, suffix):
    image_np = np.array(image)
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

    rearranged_image = Image.fromarray(rearranged_image_np)
    rearranged_output_path = output_path + suffix
    rearranged_image.save(rearranged_output_path)
    print(f"Rearranged image saved at: {rearranged_output_path}")
    return rearranged_image

def create_repeated_image(image, h_border, w_border, output_path, suffix):
    image_np = np.array(image)
    h, w, c = image_np.shape
    repeated_image_np = np.ones((h * 2 + h_border*2, w * 2 + w_border*2, c), dtype=image_np.dtype)*255

    repeated_image_np[:h, :w] = image_np
    repeated_image_np[:h, w + w_border*2:] = image_np
    repeated_image_np[h + h_border*2:, :w] = image_np
    repeated_image_np[h + h_border*2:, w + w_border*2:] = image_np

    repeated_image = Image.fromarray(repeated_image_np)
    repeated_output_path = output_path + suffix
    repeated_image.save(repeated_output_path)
    print(f"Repeated image saved at: {repeated_output_path}")
    return repeated_image

def create_mask(h, w, h_border, w_border, output_path):
    mask = np.zeros((h * 2 + h_border*2, w * 2 + w_border*2), dtype=np.uint8)
    mask[h:h + h_border*2, :] = 255
    mask[:, w:w + w_border*2] = 255

    mask_image = Image.fromarray(mask)
    mask_path = output_path + "_3_mask.png"
    mask_image.save(mask_path)
    print(f"Mask image saved at: {mask_path}")
    return mask_image

def create_prompt(image, output_path):
    # from transformers import AutoProcessor, Blip2ForConditionalGeneration
    # # Load the processor and model
    # processor = AutoProcessor.from_pretrained("./models/blip2-opt-2.7b")
    # model = Blip2ForConditionalGeneration.from_pretrained("./models/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda")
    
    # # Preprocess the image
    # inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
    
    # # Generate the description
    # generated_ids = model.generate(**inputs, max_new_tokens=20)
    # description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
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
    
    with open(output_path + "_description.txt", "w") as f:
        f.write(description)
    print(f"Description saved at: {output_path + '_description.txt'}")

    return description

def flux_fill(image, mask, prompt, h, w, output_path):
    pipe = FluxFillPipeline.from_pretrained("./models/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")
    flux_image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=h,
        width=w,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    flux_output_path = output_path + "_4_flux.png"
    flux_image.save(flux_output_path)
    print(f"Flux image saved at: {flux_output_path}")
    return flux_image

def create_seamless_check(image, suffix, output_path, repeat_count=3):
    resized_image = image.resize((300, 300))
    repeated_image = Image.new('RGB', (300 * repeat_count, 300 * repeat_count))

    for i in range(repeat_count):
        for j in range(repeat_count):
            repeated_image.paste(resized_image, (i * 300, j * 300))

    seamless_output_path = output_path + suffix
    repeated_image.save(seamless_output_path)
    print(f"Seamless check image saved at: {seamless_output_path}")
    return repeated_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image or all images in a folder to create seamless textures.")
    parser.add_argument("--image_path", type=str, help="Path to the input image or folder containing images", default="inputs/Texture/c0.jpg")
    args = parser.parse_args()

    def process_image(image_path):
        original_image = load_image(image_path)
        output_path = os.path.join("outputs/Flux_fill_SeamlessTexture", os.path.basename(image_path).split(".")[0])
        original_image.save(output_path + "_original.png")
        image = original_image.crop((100, 100, 900, 900))
        image.save(output_path + "_0_cropped.png")

        w, h = image.size
        w_border, h_border = w // 10, h // 10
        rearranged_image = create_repeated_image(image, h_border, w_border, output_path, suffix="_2_rearranged.png")
        mask = create_mask(h, w, h_border, w_border, output_path)
        prompt = create_prompt(image, output_path)
        flux_image = flux_fill(rearranged_image, mask, prompt, (h+h_border)*2, (w+w_border)*2, output_path)
        cropped_image = flux_image.crop((w//2, h//2, w//2*3+w_border*2, h//2*3+h_border*2))
        cropped_image.save(output_path + "_5_cropped.png")
        final_image = rearrange_quadrants(cropped_image, output_path, suffix="_6_final.png")

        create_seamless_check(original_image, "_7_original_seamless.png", output_path)
        create_seamless_check(image, "_8_cropped_seamless.png", output_path)
        create_seamless_check(final_image, "_9_final_seamless.png", output_path)

    if os.path.isdir(args.image_path):
        for filename in os.listdir(args.image_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                process_image(os.path.join(args.image_path, filename))
    else:
        process_image(args.image_path)