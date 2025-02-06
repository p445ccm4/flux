import os
import moviepy
from rife_model import load_rife_model, rife_inference_with_latents
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_video
import torch
import torchvision

def sort_by_startint(a):
    return int(a.split("_")[0])

def split_sentence(prompt, max_length=100):
    result = []
    while len(prompt) > max_length:
        split_index = prompt.rfind(' ', 0, max_length)
        if split_index == -1:
            split_index = max_length
        result.append(prompt[:split_index])
        prompt = prompt[split_index + 1:]
    result.append(prompt)
    return '\n'.join(result)

# load
device = 'cuda'
frame_interpolation_model = load_rife_model("./models/rife")

working_dir = "outputs/HunYuan/20250205"
# vid_list = [f for f in os.listdir(working_dir) if f.endswith("_captioned.mp4")]
# vid_list = sorted(vid_list, key=sort_by_startint)

# for vid in vid_list:
for i in range(0, 60):
    # Interpolation ###################################
    input_video_path = os.path.join(working_dir, f"{i}.mp4")
    diffusion_result, _, _ = torchvision.io.read_video(input_video_path)

    # preprocess
    diffusion_result = diffusion_result.to(device) / 255.0
    diffusion_result = torch.permute(diffusion_result, (0, 3, 1, 2))
    diffusion_result = torch.unsqueeze(diffusion_result, 0)

    # inference
    diffusion_result = rife_inference_with_latents(frame_interpolation_model, diffusion_result) # 5->10
    diffusion_result = rife_inference_with_latents(frame_interpolation_model, diffusion_result) # 10->20

    # postprocess
    diffusion_result = VaeImageProcessor.pt_to_numpy(diffusion_result[0])

    # export
    save_video_path = os.path.join(input_video_path.replace('.mp4', '_interpolated.mp4'))
    export_to_video(diffusion_result, save_video_path, fps=20)
    ####################################################

    # Audio and Caption ###########################################
    # Load prompts
    with open("inputs/peaks_vs_ages_rewrite.txt") as f:
        lines = f.readlines()
        sentences = [line.strip() for line in lines if line != "\n" and not line.startswith("(\"")]
        prompts = [line.strip()[2:-2] for line in lines if line != "\n" and line.startswith("(\"")]
        assert len(sentences) == len(prompts), f"sentences: {len(sentences)}, prompts: {len(prompts)}"

    # Load audios
    audio_clip = moviepy.AudioFileClip(os.path.join(working_dir, f"{i}.mp3"))
    
    # Put caption on each video
    video_path = os.path.join(working_dir, f"{i}_interpolated.mp4")
    output_path = os.path.join(working_dir, f"{i}_captioned.mp4")
    video_clip = moviepy.VideoFileClip(video_path)
    text_clip = (
        moviepy.TextClip(
            font="Chilanka-Regular", 
            text=split_sentence(sentences[i], max_length=50), 
            method="caption",
            size=(video_clip.w, None),
            font_size=30, 
            color="white", 
            stroke_color="black", 
            stroke_width=1,
            text_align="center",
            )
            .with_position(("center", "bottom"))
            .with_duration(video_clip.duration)
    )
    video_text_clip = moviepy.CompositeVideoClip(clips=[video_clip, text_clip])
    video_text_audio_clip = video_text_clip.with_audio(audio_clip)
    video_text_audio_clip.write_videofile(output_path)
    ####################################################