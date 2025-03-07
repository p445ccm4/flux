import torch
from diffusers import FluxPipeline

class Flux():
    def __init__(self):
        self.pipe = FluxPipeline.from_pretrained(
            "./models/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def gen_image(self, prompt):
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=40, 
            guidance_scale=30.0,
            width=1024, 
            height=1024,
        ).images[0]

        return image
    
if __name__ == "__main__":
    flux = Flux()
    image = flux.gen_image("A heart on the left and a brain on the right. A iPhone style toggle switch in the middle. The switch is set to the heart side. Flat and minimalistic style.")
    image.save("output.jpg")