from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", local_files_only=True)
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "With animation style, An east-asian young, handsome swordman, wears his sword gleams very brilliantly. he ride a gallant white steed. His speed is like shadow or swift meteor."

# base output latent, so output is not pil
images_1 = pipe(prompt=prompt, output_type="latent", num_inference_steps=200, num_images_per_prompt=4).images
print(f"sta_1 len:{len(images_1)}")
# for i, img in enumerate(images_1):
#     pil_images_1 = pt_to_pil(img)
#     pil_images_1.save(f"./demo/sta_{i}_1.png")
# images_1[0].save(f"./demo/xl_1.png")

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", local_files_only=True)
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

images_2 = pipe(prompt=prompt, image=images_1, num_inference_steps=200, num_images_per_prompt=4).images
for i, img in enumerate(images_2):
    # img = pt_to_pil(img)
    img.save(f"./demo/xl_{i}_2.png")
# images_2[0].save(f"./demo/xl_2.png")