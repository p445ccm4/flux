import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import ImageOps, Image
import numpy as np
import os

def pad_image(image, w_border, h_border, output_path):
    padded_image = ImageOps.expand(image, border=(w_border, h_border), fill="black")
    padded_output_path = output_path + "_1_padded_image.jpg"
    padded_image.save(padded_output_path)
    print(f"Padded image saved at: {padded_output_path}")
    return padded_image

def rearrange_quadrants(padded_image_np, output_path, suffix):
    h_mid, w_mid = padded_image_np.shape[0] // 2, padded_image_np.shape[1] // 2
    top_left = padded_image_np[:h_mid, :w_mid]
    top_right = padded_image_np[:h_mid, w_mid:]
    bottom_left = padded_image_np[h_mid:, :w_mid]
    bottom_right = padded_image_np[h_mid:, w_mid:]

    rearranged_image_np = np.zeros_like(padded_image_np)
    rearranged_image_np[:h_mid, :w_mid] = bottom_right
    rearranged_image_np[:h_mid, w_mid:] = bottom_left
    rearranged_image_np[h_mid:, :w_mid] = top_right
    rearranged_image_np[h_mid:, w_mid:] = top_left

    rearranged_image = Image.fromarray(rearranged_image_np)
    rearranged_output_path = output_path + suffix
    rearranged_image.save(rearranged_output_path)
    print(f"Rearranged image saved at: {rearranged_output_path}")
    return rearranged_image

def create_mask(padded_image_np, h, w, h_border, w_border, output_path):
    mask = np.zeros_like(padded_image_np, dtype=np.uint8)
    mask[h // 2:h // 2 + h_border * 2, :] = 255
    mask[:, w // 2:w // 2 + w_border * 2] = 255

    mask_image = Image.fromarray(mask)
    mask_path = output_path + "_3_mask.jpg"
    mask_image.save(mask_path)
    print(f"Mask image saved at: {mask_path}")
    return mask_image

def flux_fill(rearranged_image, mask, h, w, h_border, w_border, output_path):
    pipe = FluxFillPipeline.from_pretrained("./models/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")
    flux_image = pipe(
        prompt="a texture with consistent pattern",
        image=rearranged_image,
        mask_image=mask,
        height=h + h_border * 2,
        width=w + w_border * 2,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    flux_output_path = output_path + "_4_flux.jpg"
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
    image_path = "inputs/Texture/Bardiglio_Chiaro_1K.jpg"
    original_image = load_image(image_path)
    output_path = os.path.join("outputs/Flux_fill_SeamlessTexture", os.path.basename(image_path).replace(".jpg", ""))
    original_image.save(output_path + "_0_original.png")
    original_repeated_image = create_seamless_check(original_image, "_7_original_seamless.jpg", output_path)
    image = original_image.crop((200, 200, 800, 800))
    image.save(output_path + "_0_cropped.png")

    w, h = image.size
    w_border, h_border = w // 10, h // 10
    padded_image = pad_image(image, w_border, h_border, output_path)
    padded_image_np = np.array(padded_image)
    rearranged_image = rearrange_quadrants(padded_image_np, output_path, suffix="_2_rearranged.jpg")
    mask = create_mask(padded_image_np, h, w, h_border, w_border, output_path)
    flux_image = flux_fill(rearranged_image, mask, h, w, h_border, w_border, output_path)
    flux_image_np = np.array(flux_image)
    final_image = rearrange_quadrants(flux_image_np, output_path, suffix="_5_final.jpg")
    final_repeated_image = create_seamless_check(final_image, "_8_final_seamless.jpg", output_path)