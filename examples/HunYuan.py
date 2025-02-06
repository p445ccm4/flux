import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "./models/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
pipe.vae.enable_tiling()
pipe.to("cuda")

output = pipe(
    prompt="Donald Trump eating Chinese food in a Chinese restaurant",
    height=720,
    width=1280,
    num_frames=61,
    num_inference_steps=50,
).frames[0]
export_to_video(output, "hunyuan.mp4", fps=15)