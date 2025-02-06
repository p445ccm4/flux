import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from diffusers.utils import export_to_video
import moviepy
from gtts import gTTS
import argparse

parser = argparse.ArgumentParser(description="Generate videos from text prompts.")
parser.add_argument("-i", type=int, required=True, help="Index of the prompt to process")
args = parser.parse_args()
i = args.i

# Load txt2vid model
model_id = "./models/HunyuanVideo"
# quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16#, quantization_config=quant_config
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload() 

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

# Load prompts
with open("inputs/peaks_vs_ages_rewrite.txt") as f:
    lines = f.readlines()
    sentences = [line.strip() for line in lines if line != "\n" and not line.startswith("(\"")]
    prompts = [line.strip()[2:-2] for line in lines if line != "\n" and line.startswith("(\"")]
    assert len(sentences) == len(prompts), f"sentences: {len(sentences)}, prompts: {len(prompts)}"

# clips = []
prompt = prompts[i]
sentence = sentences[i]
print(f"i: {i}\nsentence: {sentence}\nprompt: {prompt}\n")

# Generate audios
audio = gTTS(sentence, lang="en", tld="us") # leng="yue" for Cantonese
audio.save(f"outputs/HunYuan/{i}.mp3")
audio_clip = moviepy.AudioFileClip(f"outputs/HunYuan/{i}.mp3")

fps = 5
# Generate videos
num_frames = int(audio_clip.duration * fps * 4 // 4 + 1)
print(f"num_frames: {num_frames}")
output = pipe(
    prompt=f"{prompt}",
    height=720,
    width=1280,
    num_frames=num_frames,
    num_inference_steps=40,
).frames[0]
export_to_video(output, f"outputs/HunYuan/{i}.mp4", fps=fps)

# Put caption on each video
video_path = f"outputs/HunYuan/{i}.mp4"
output_path = f"outputs/HunYuan/{i}_captioned.mp4"
video_clip = moviepy.VideoFileClip(video_path)
text_clip = (
       moviepy.TextClip(
           font="Chilanka-Regular", 
           text=split_sentence(sentence, max_length=50), 
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
# clips.append(video_text_audio_clip)

# # Concatenate all clips
# concat_clip = moviepy.concatenate_videoclips(clips)
# concat_clip.write_videofile("outputs/HunYuan/concat_captioned.mp4")
