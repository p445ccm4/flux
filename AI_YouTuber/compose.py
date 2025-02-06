import moviepy
import os

def sort_by_startint(a):
    return int(a.split("_")[0])

working_dir = "outputs/HunYuan/20250205"
vid_list = [f for f in os.listdir(working_dir) if f.endswith("_captioned.mp4")]
vid_list = sorted(vid_list, key=sort_by_startint)
clips = [moviepy.VideoFileClip(os.path.join(working_dir, f)) for f in vid_list]

print(f"clips: {vid_list}")

# Concatenate all clips
concat_clip = moviepy.concatenate_videoclips(clips)
concat_clip.write_videofile(os.path.join(working_dir, "concat.mp4"))
