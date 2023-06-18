import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor, CLIPImageProcessor
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import gc, subprocess
import img_rename
from random import randrange
import json, base64


def setting(model_dic, gpu = 0):
    #GPU setting
    torch.cuda.set_device(int(gpu))
    print('loading by ', model_dic)
    
    #ckpt/anything-v4.5-vae-swapped | Azher/Anything-v4.5-vae-fp16-diffuser | Nacholmo/AbyssOrangeMix2-hard-vae-swapped 
    #Nacholmo/Counterfeit-V2.5-vae-swapped | 
    vae = AutoencoderKL.from_pretrained(f'./anime_vae', subfolder="anything")
    #vae = AutoencoderKL.from_pretrained(f'{model_dic}', subfolder="vae")
    #vae = AutoencoderKL.from_pretrained(f'./anime_vae', subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(f'{model_dic}', subfolder="text_encoder")
    
    tokenizer = CLIPTokenizer.from_pretrained(f'{model_dic}', subfolder="tokenizer")
    
    unet = UNet2DConditionModel.from_pretrained(f'{model_dic}', subfolder="unet")
   
    scheduler = PNDMScheduler.from_pretrained(f'{model_dic}', subfolder="scheduler")

    feature_extractor = CLIPFeatureExtractor.from_pretrained(f'{model_dic}', subfolder="feature_extractor")
    
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=f'{model_dic}',
                                               vae=vae,
                                               text_encoder=text_encoder,
                                               tokenizer=tokenizer,
                                               unet=unet,
                                               scheduler=scheduler,
                                               safety_checker=None,
                                               feature_extractor=feature_extractor,
                                               torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    
    return pipe
def txt2img(model_dic, pipe, prompt, user_id, json_object):
   #height, width
    gc.collect()
    torch.cuda.empty_cache()
    
    #seed
    seed = randrange(300000000)
    prompt = prompt + ', ((best quality)), ((masterpiece)), (best illustration), ((cinematic light)), hyper detail'
    negative_prompt = '(monochrome:1.3), (oversaturated:1.3), ((hands)), (worst quality, low quality:1.4), multiple views, 3d render, cartoon, ((blurry)), (text, signature, artist name, watermark), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (too many fingers), (((long neck))), (((mutation))), (((deformed))), ((ugly))'
    print('prompt : ', prompt)

    num_samples = 4
    guidance_scale = 6.5
    num_inference_steps = 100
    height = 512 
    width = 512 

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed)
        ).images
    
    
    file_path = img_rename.make_dic(user_id)
    index = 0
    
    for img in images:
        img.save(file_path + f'/{index}.{seed}.png','PNG')
        with open(file_path + f'/{index}.{seed}.png', 'rb') as one:
                    image_data = one.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        json_object[f'image{index}'] = {
                "image" : encoded_image,
            }
        index = index + 1
        
    return json_object

def main_process(prompt, model_dic,user_id, json_object, gpu):
    
    pipe = setting(model_dic, gpu)
    json_object= txt2img(model_dic,pipe, prompt, user_id, json_object)
    return json_object