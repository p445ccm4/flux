import subprocess
from gtts import gTTS
import argparse
import os

class AudioGenerator:
    def __init__(self, output_dir="outputs/HunYuan"):
        self.output_dir = output_dir

    def generate_audio(self, sentence, index):
        temp_audio_path = f"{self.output_dir}/{index}_temp.mp3"
        audio = gTTS(sentence, lang="en", tld="us") # lang="yue" for Cantonese, (lang="en", tld="us") for US English
        audio.save(temp_audio_path)
        return temp_audio_path

    def speed_up_audio(self, temp_audio_path, speed_factor=1.5):
        audio_path = temp_audio_path.replace("_temp", "")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        subprocess.call([
            "ffmpeg",
            "-i", temp_audio_path,
            "-filter:a", f"atempo={speed_factor}",
            audio_path
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos from text prompts.")
    parser.add_argument("-i", type=int, required=True, help="Index of the prompt to process")
    parser.add_argument("-c", type=str, required=True, help="Cantonese caption")
    parser.add_argument("-o", type=str, default="outputs/HunYuan", help="Output directory for audio files")
    args = parser.parse_args()

    generator = AudioGenerator(args.o)
    temp_audio_path = generator.generate_audio(args.c, args.i)
    generator.speed_up_audio(temp_audio_path)
