from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16, local_files_only=True)
pipe.enable_model_cpu_offload()

# prompt = 'A Chinese young , handsome swordman named Li Bai, The blade of Wugou sword was as bright as frost and snow. The silver saddle lighted the white horse, Rustling like a meteor when running'
prompt = "A knight-errant carries a sword which is gleams brilliantly, as if it is forged from frost and snow, its razor sharp edge reflecting the cold yet dazzling light. This illumination is further mirrored in the polished silver saddle, which in turn casts a gleaming hue onto the white steed he rides, creating an alluring dance of lights between horse and rider. The sight is nothing short of resplendent, like a waltz of celestial bodies in the night sky. And when he urges his steed into a gallop, it's akin to a meteor streaking across the heavens, casting a breathtaking scene of both strength and beauty. The vision of the heroic figure in flight is so swift, so transient, that it feels as though it is both of this world and not, a fleeting echo of a comet's passing, leaving awe and wonderment in its wake."
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