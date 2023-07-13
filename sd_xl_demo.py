from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", local_files_only=True)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "With animation style, An east-asian young, handsome swordman, wears his sword gleams very brilliantly. he ride a gallant white steed. His speed is like shadow or swift meteor."

images_1 = pipe(prompt=prompt, output_type="latent").images
pil_images_1 = pt_to_pil(images_1)
pil_images_1.save(f"./demo/xl_1.png")

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", local_files_only=True)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

images_2 = pipe(prompt=prompt, image=images_1).images
for i, img in enumerate(pil_images_2):
    img = pt_to_pil(img)
    img.save(f"./demo/xl_2_{i}.png")