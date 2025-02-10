import argparse
import os
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

def add_background_music(video_path, music_path, output_path):
    """
    Adds background music to a video using MoviePy.

    Args:
        video_path (str): Path to the video file.
        music_path (str): Path to the audio file (background music).
        output_path (str): Path to save the video with background music.
    """
    try:
        video_clip = VideoFileClip(video_path).with_volume_scaled(1.5)
        audio_clip = AudioFileClip(music_path).with_volume_scaled(0.3)

        # Ensure the music is as long as the video
        start = audio_clip.duration/2 - video_clip.duration/2
        end = audio_clip.duration/2 + video_clip.duration/2
        audio_clip = audio_clip.subclipped(start, end)

        # Combine video and audio
        final_audio = CompositeAudioClip([video_clip.audio, audio_clip])
        video_clip.audio = final_audio

        # Write the result to a file
        video_clip.write_videofile(output_path, logger=None)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add background music to a video.")
    parser.add_argument("--input_video_path", help="Path to the video file")
    parser.add_argument("--music_path", help="Path to the audio file (background music)")
    parser.add_argument("--output_video_path", help="Path to save the video with background music")

    args = parser.parse_args()

    add_background_music(args.input_video_path, args.music_path, args.output_video_path)
    print(f"Successfully added background music to {args.input_video_path} and saved as {args.output_video_path}")

    # Delete the original video
    os.remove(args.input_video_path)