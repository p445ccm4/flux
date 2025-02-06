import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
import moviepy

model_id = "./models/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

def split_prompt(prompt, max_length=100):
    result = []
    while len(prompt) > max_length:
        split_index = prompt.rfind(' ', 0, max_length)
        if split_index == -1:
            split_index = max_length
        result.append(prompt[:split_index])
        prompt = prompt[split_index + 1:]
    result.append(prompt)
    return '\n'.join(result)

with open("inputs/peaks_vs_ages.txt") as f:
    prompts = [line.strip() for line in f.readlines() if line != "\n"]

clips = []
for i, prompt in enumerate(prompts):
    if i > 11:
        break
    print(f"prompt: {prompt}, i: {i}")
    print(f"splitted prompt: {split_prompt(prompt)}")

    # Generate videos
    output = pipe(
        prompt=prompt,
        height=720,
        width=1280,
        num_frames=61,
        num_inference_steps=50,
    ).frames[0]
    export_to_video(output, f"outputs/HunYuan/{i}.mp4", fps=15)

    # Put caption on each video
    video_path = f"outputs/HunYuan/{i}.mp4"
    output_path = f"outputs/HunYuan/{i}_captioned.mp4"
    video_clip = moviepy.VideoFileClip(output_path)
    text_clip = (
        moviepy.TextClip(font="Chilanka-Regular", text=split_prompt(prompt), font_size=20, color="white")
        .with_position(("center", "bottom"))
        .with_duration(video_clip.duration)
    )
    video_text_clip = moviepy.CompositeVideoClip(clips=[video_clip, text_clip])
    video_text_clip.write_videofile(output_path)
    clips.append(video_text_clip)

concat_clip = moviepy.concatenate_videoclips(clips)
concat_clip.write_videofile("outputs/HunYuan/concat_captioned.mp4")
