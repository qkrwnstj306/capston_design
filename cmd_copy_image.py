import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor, CLIPImageProcessor
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import gc, subprocess
import image_rename_for_copy
from random import randrange
import json, base64


def setting(model_dic, gpu = 0):
    #GPU setting
    torch.cuda.set_device(int(gpu))
    
    #vae = AutoencoderKL.from_pretrained(f'./civit_model_diffusers/{model_dic}', subfolder="vae")
    vae = AutoencoderKL.from_pretrained(f'./anime_vae', subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(f'./civit_model_diffusers/{model_dic}', subfolder="text_encoder")
    
    tokenizer = CLIPTokenizer.from_pretrained(f'./civit_model_diffusers/{model_dic}', subfolder="tokenizer")
    
    unet = UNet2DConditionModel.from_pretrained(f'./civit_model_diffusers/{model_dic}', subfolder="unet")
   
    scheduler = PNDMScheduler.from_pretrained(f'./civit_model_diffusers/{model_dic}', subfolder="scheduler")

    feature_extractor = CLIPFeatureExtractor.from_pretrained(f'./civit_model_diffusers/{model_dic}', subfolder="feature_extractor")
    #CLIPFeatureExtractor.from_pretrained(f'./civit_model_diffusers/{model_dic}', subfolder="feature_extractor")
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=f'./civit_model_diffusers/{model_dic}',
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

def txt2img(model_dic, pipe, prompt, seed, user_id, index):
   #height, width
    gc.collect()
    torch.cuda.empty_cache()
    
    # rev
    if model_dic == 'revAnimate':
        prompt = prompt + f', ((solo)), ((masterpiece)), (best illustration), ((cinematic light)), hyper detail, (extremely detailed CG unity 8k wallpaper, masterpiece, ultra-detailed)'
        negative_prompt = '((hands)), (((mutation))), (((deformed))), ((ugly)), (worst quality, low quality:1.4),((blurry)), ((horns)), bright, (monochrome:1.3), (oversaturated:1.3), bad hands, lowres, 3d render, cartoon, long body, ((blurry)), duplicate, ((duplicate body parts)), ugly, poor quality, low quality, out of frame'
    #pastel
    if model_dic == 'pastelboys':
        prompt = prompt + f', ((masterpiece)), best quality, ((solo)), (handsome), highly detailed face, colorful Fantastic light and shadow, Scenery, mature,'
        negative_prompt = '((hands)), (worst quality, low quality:1.4), multiple views, 3d render, cartoon, ((blurry)), (text, signature, artist name, watermark), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (too many fingers), (((long neck))), (((mutation))), (((deformed))), ((ugly))'
    #dreamshaper
    if model_dic == 'dreamshaper':
        prompt = prompt + f', ((solo)), (masterpiece), (extremely intricate:1.3), (realistic), detailed, dramatic, award winning, matte drawing, cinematic lighting, octane render, unreal engine'
        negative_prompt = '((hands)), (worst quality, low quality:1.4), canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render'
    #meinamix
    if model_dic == 'meinamix' :
        prompt = prompt + f', ((solo)), (masterpiece), glowing, sidelighting, wallpaper, no hands, ((wearing clothes)), detailed'
        negative_prompt = '((hands)), (worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic), ((blurry)), (text, signature, artist name, watermark), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (too many fingers), (((long neck))), (((mutation))), (((deformed))), ((ugly))'
    #Counterfeit
    if model_dic == 'Counterfeit' :
        prompt = prompt + f', ((masterpiece)), best quality, ((solo)), ((masterpiece,best quality)), blurry background, outdoors'
        negative_prompt = '((hands)), (worst quality, low quality:1.4), multiple views, 3d render, cartoon, ((blurry)), (text, signature, artist name, watermark), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (too many fingers), (((long neck))), (((mutation))), (((deformed))), ((ugly))'
    
    num_samples = 1
    guidance_scale = 8
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
    
    
    file_path = image_rename_for_copy.make_dic(user_id)
    
    for img in images:
        img.save(file_path + f'/{index}.{seed}.png','PNG')
        

def main_process(prompt, model_dic,user_id, seed, index, gpu):
    #model_dic : pastelboys or revAnimate etc.
    
    pipe = setting(model_dic, gpu)
    txt2img(model_dic,pipe, prompt, seed, user_id, index)
    