from diffusers import MochiPipeline
from diffusers.utils import export_to_video
import torch
import json

class Text2Video:
	def __init__(self):
		self.pipe = MochiPipeline.from_pretrained("./models/mochi-1-preview")

		# Enable memory savings
		self.pipe.enable_model_cpu_offload()
		self.pipe.enable_vae_tiling()

	def gen_video(self, text):
		with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
			self.frames = self.pipe(
				text, 
				num_inference_steps=50, 
				guidance_scale=4.5, 
				height=480, 
				width=848, 
				num_frames=121
				).frames[0]

	def save_video(self, path):
		export_to_video(self.frames, path, fps=15)

if __name__ == '__main__':
	# with open('inputs/AI_shorts_prompt.json', 'r') as file:
	# 	data = json.load(file)
	# 	lines = [item['video_prompt'] for item in data]
	lines = ['Art Deco woman, 1920s glamour style, flapper dress, elegant jewelry, golden lighting, geometric patterns, luxurious setting, classic Hollywood aesthetic, stylized portraiture, vintage beauty reimagined by AI']
	
	text_2_video = Text2Video()
	for i, line in enumerate(lines):
		print(line)
		text_2_video.gen_video(line)
		text_2_video.save_video(f'outputs/Mochi.mp4')