from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16, local_files_only=True)
pipe.enable_model_cpu_offload()

# prompt = 'A Chinese young, handsome swordman named Li Bai, The blade of Wugou sword was as bright as frost and snow. The silver saddle lighted the white horse, Rustling like a meteor when running'
# prompt = "A Chinese young, handsome swordman, wears his sword gleams very brilliantly, shows a heroic spirit. he ride a gallant white steed. When they gallop, their combined force is like a swift meteor, streaking across the vast expanse, a poignant picture of might and elegance."
prompt = "A Chinese young, handsome swordman, wears his sword gleams very brilliantly, shows a heroic spirit. he ride a gallant white steed. His speed is like shadow or swift meteor."
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
    "watermarker": pipe.watermarker,
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