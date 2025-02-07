from moviepy import VideoFileClip, AudioFileClip
import argparse

def combine_video_audio(video_path, audio_path, output_path):
    """
    Combines a video with an audio file using MoviePy.

    Args:
        video_path (str): Path to the video file.
        audio_path (str): Path to the audio file.
        output_path (str): Path to save the combined video.
    """
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        print(f"Successfully combined video and audio. Output saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine video and audio files.")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument("output_path", help="Path to save the combined video")

    args = parser.parse_args()

    combine_video_audio(args.video_path, args.audio_path, args.output_path)