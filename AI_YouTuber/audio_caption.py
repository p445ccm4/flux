import argparse
import moviepy

class VideoCaptioner:
    def __init__(self, audio_path, caption, input_video_path, output_video_path):
        self.audio_path = audio_path
        self.caption = caption
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

    def add_audio_and_caption(self):
        # Load audio
        audio_clip = moviepy.AudioFileClip(self.audio_path)
        
        # Load video
        video_clip = moviepy.VideoFileClip(self.input_video_path)
        
        # Create text clip
        text_clip = (
            moviepy.TextClip(
                font="Chilanka-Regular", 
                text=self.caption, 
                method="caption",
                size=(video_clip.w, None),
                font_size=80, 
                color="white", 
                bg_color=None,
                stroke_color="black", 
                stroke_width=1,
                text_align="center",
            )
            .with_position(("center", 900))
            .with_duration(video_clip.duration)
        )
        
        # Combine video, text, and audio
        video_text_clip = moviepy.CompositeVideoClip(clips=[video_clip, text_clip])
        video_text_audio_clip = video_text_clip.with_audio(audio_clip)
        
        # Write the output video file
        video_text_audio_clip.write_videofile(self.output_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add audio and caption to a video.")
    parser.add_argument("--audio_path", type=str, help="Path to the audio file")
    parser.add_argument("--caption", type=str, help="Caption text to add to the video")
    parser.add_argument("--input_video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output_video_path", type=str, help="Path to the output video file")
    
    args = parser.parse_args()
    
    captioner = VideoCaptioner(
        audio_path=args.audio_path,
        caption=args.caption,
        input_video_path=args.input_video_path,
        output_video_path=args.output_video_path
    )
    
    captioner.add_audio_and_caption()