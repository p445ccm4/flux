from diffusers import MochiPipeline, MochiTransformer3DModel
from diffusers.utils import export_to_video
import torch

class Text2Video:
	def __init__(self):
		# transformer = MochiTransformer3DModel.from_pretrained("imnotednamode/mochi-1-preview-mix-nf4", torch_dtype=torch.bfloat16)
		transformer = MochiTransformer3DModel.from_pretrained("imnotednamode/mochi-1-preview-mix-nf4-small", torch_dtype=torch.bfloat16)
		self.pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", revision="refs/pr/18", torch_dtype=torch.bfloat16, transformer=transformer)
		self.pipe.enable_model_cpu_offload()
		self.pipe.enable_vae_tiling()

	def gen_video(self, text):
		self.frames = self.pipe(
			text, 
			num_inference_steps=50, 
			guidance_scale=4.5, 
			height=480, 
			width=848, 
			num_frames=15
			).frames[0]

	def save_video(self, path):
		export_to_video(self.frames, path, fps=15)

if __name__ == '__main__':
	with open("txt", "r") as f:
		lines = f.readlines()
	
	text_2_video = Text2Video()
	for i, line in enumerate(lines):
		print(line)
		text_2_video.gen_video(line)
		text_2_video.save_video(f'outputs/Mochi/{i}.mp4')