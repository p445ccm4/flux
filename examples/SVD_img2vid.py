import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "./models/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
# pipe.enable_model_cpu_offload()
pipe.to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# Load the conditioning image
image = load_image("inputs/meme material/joey2.png")
h, w = image.size
image = image.resize((1024, 576))

frames = pipe(image, decode_chunk_size=8, motion_bucket_id=180, noise_aug_strength=0.0, num_inference_steps=50).frames[0]
export_to_video(frames, "outputs/meme_materials/temp.mp4", fps=7)

frames = pipe(image, decode_chunk_size=8, motion_bucket_id=180, noise_aug_strength=0.0, num_inference_steps=50).frames[0]
export_to_video(frames, "outputs/meme_materials/temp2.mp4", fps=7)

# resize the video back to original size
import moviepy as mp
clip = mp.VideoFileClip("outputs/meme_materials/temp.mp4")
clip2 = mp.VideoFileClip("outputs/meme_materials/temp2.mp4")
clip_resized = clip.with_effects([mp.vfx.Resize((h, w))])
clip2_resized = clip2.with_effects([mp.vfx.Resize((h, w))])
concat_clip = mp.concatenate_videoclips([clip_resized, clip2_resized])
concat_clip.write_videofile("outputs/meme_materials/joey2_final.mp4")
