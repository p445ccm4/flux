import moviepy
from gtts import gTTS


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

# Load prompts
with open("inputs/peaks_vs_ages_prompt.txt") as f:
    prompts = [line.strip() for line in f.readlines() if line != "\n"]



clips = []
for i, prompt in enumerate(prompts):
    if i > 5:
        continue
    print(f"i: {i}\nsplitted prompt: {split_prompt(prompt)}\n")
    
    # Generate audios
    audio = gTTS(prompt, lang="en") # leng="yue" for Cantonese
    audio.save(f"outputs/HunYuan/{i}.mp3")
    audio_clip = moviepy.AudioFileClip(f"outputs/HunYuan/{i}.mp3")

    # Put caption on each video
    video_path = f"outputs/HunYuan/{i}.mp4"
    output_path = f"outputs/HunYuan/{i}_captioned.mp4"
    video_clip = moviepy.VideoFileClip(video_path)
    text_clip = (
        moviepy.TextClip(font="Chilanka-Regular", text=split_prompt(prompt), font_size=20, color="white")
        .with_position(("center", "bottom"))
        .with_duration(video_clip.duration)
    )
    video_text_clip = moviepy.CompositeVideoClip(clips=[video_clip, text_clip])
    video_text_audio_clip = video_text_clip.with_audio(audio_clip)
    video_text_audio_clip.write_videofile(output_path)
    clips.append(video_text_audio_clip)

# Concatenate all clips
concat_clip = moviepy.concatenate_videoclips(clips)
concat_clip.write_videofile("outputs/HunYuan/concat_captioned.mp4")
