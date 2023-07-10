from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16, local_files_only=True)
pipe.enable_model_cpu_offload()

# prompt = "With animation style, An east-asian young, handsome swordman, wears his sword gleams very brilliantly. he ride a gallant white steed. His speed is like shadow or swift meteor."
prompt = "1 girl, beautiful Korean girl, (Cute Loose Bob hair), (wearing a cropped hoodie, capri sweatpants:1. 5), (hands in pockets:1. 5), (small breasts:1. 3), (eyelashes:1. 2), beautiful detailed eyes, symmetrical eyes, (detailed face), immersive background, volumetric haze, global illumination, soft lighting, (flowing hair), (bright smile), natural lighting, (realistic:1. 5), (lifelike:1. 4), (4k, digital art, masterpiece), High detail digital painting, realistic, (top quality), (soft shadows), (best character art), ultra high resolution, highly detailed digital artwork, physically-based rendering, realism with an artistic touch, vibrant colors, f2. 2 lens, soft palette, natural beauty."
prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt").images

# save intermediate image
pil_image = pt_to_pil(image)
pil_image[0].save("./demo/if_stage_1.png")

super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16, local_files_only=True)
super_res_1_pipe.enable_model_cpu_offload()

image = super_res_1_pipe(image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt").images

# save intermediate image
pil_image = pt_to_pil(image)
pil_image[0].save("./demo/if_stage_2.png")

# numpy_img = pt_to_numpy(image)

# if model safety_checker need numpy input, but sd-upscaler's input is pil_image to tensor
safety_modules = {
    "feature_extractor": pipe.feature_extractor,
    # "safety_checker": super_res_1_pipe.safety_checker,
    # "watermarker": pipe.watermarker,
}
super_res_2_pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16, local_files_only=True
)
super_res_2_pipe.enable_model_cpu_offload()

# pass image will cause warning, use pil_image instead
image = super_res_2_pipe(
    prompt=prompt,
    image=pil_image,
).images
image[0].save("./demo/if_stage_3.png")