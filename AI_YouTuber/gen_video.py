import torch
from diffusers import HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from diffusers.utils import export_to_video
import moviepy
import argparse

class VideoGenerator:
    def __init__(self, output_dir="outputs/HunYuan"):
        model_id = "./models/HunyuanVideo"
        self.output_dir = output_dir
        self.transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        self.pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=self.transformer, torch_dtype=torch.float16)
        self.pipe.vae.enable_tiling()
        self.pipe.enable_model_cpu_offload()

    def generate_video(self, prompt, index, fps=5, num_frames=None):
        if num_frames is None:
            audio_path = f"{self.output_dir}/{index}.mp3"
            audio_clip = moviepy.AudioFileClip(audio_path)
            num_frames = audio_clip.duration * fps
        num_frames = int(num_frames // 4 * 4 + 1)
        print(f"num_frames: {num_frames}")
        output = self.pipe(
            prompt=prompt,
            height=1280,
            width=720,
            num_frames=num_frames,
            num_inference_steps=40,
        ).frames[0]
        export_to_video(output, f"{self.output_dir}/{index}.mp4", fps=fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos from text prompts.")
    parser.add_argument("-i", type=int, required=True, help="Index of the prompt to process")
    parser.add_argument("-e", type=str, required=True, help="English prompt")
    parser.add_argument("-o", type=str, default="outputs/HunYuan", help="Output directory")
    parser.add_argument("-n", "--num_frames", type=int, default=None, help="Number of frames to generate")
    args = parser.parse_args()

    generator = VideoGenerator(output_dir=args.o)
    generator.generate_video(args.e, args.i, num_frames=args.num_frames)
