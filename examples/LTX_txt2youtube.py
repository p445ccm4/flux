import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video
import moviepy
from gtts import gTTS
import argparse

parser = argparse.ArgumentParser(description="Generate videos from text prompts.")
parser.add_argument("-i", type=int, required=True, help="Index of the prompt to process")
args = parser.parse_args()
i = args.i

# Load txt2vid model
pipe = LTXPipeline.from_pretrained("./models/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")

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
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"


# clips = []
prompt = prompts[i]
sentence = sentences[i]
print(f"i: {i}\nsentence: {sentence}\nprompt: {prompt}\n")

# Generate audios
audio = gTTS(sentence, lang="en", tld="us") # leng="yue" for Cantonese
audio.save(f"outputs/LTX/{i}.mp3")
audio_clip = moviepy.AudioFileClip(f"outputs/LTX/{i}.mp3")

# Generate videos
num_frames = int(audio_clip.duration * 15 * 8 // 8 + 1)
print(f"num_frames: {num_frames}")
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=num_frames,
    num_inference_steps=50,
).frames[0]
export_to_video(output, f"outputs/LTX/{i}.mp4", fps=15)

# Put caption on each video
video_path = f"outputs/LTX/{i}.mp4"
output_path = f"outputs/LTX/{i}_captioned.mp4"
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
# concat_clip.write_videofile("outputs/LTX/concat_captioned.mp4")
