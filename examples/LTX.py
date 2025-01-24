import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained("./models/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A person looks worried about ageing, with a somber expression. Show the passage of time with changing clocks or fading photographs."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=704,
    height=480,
    num_frames=161,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "outputs/LTX/0.mp4", fps=24)
