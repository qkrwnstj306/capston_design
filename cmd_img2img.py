import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor, CLIPImageProcessor
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
import gc, subprocess
import img_rename
from random import randrange
import json, base64
from PIL import Image

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
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path=f'./civit_model_diffusers/{model_dic}',
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

def img2img(model_dic, pipe, prompt, user_id, index, json_object, url, seed):
   #height, width
    gc.collect()
    torch.cuda.empty_cache()
   
    # rev
    if model_dic == 'revAnimate':
        prompt = prompt + f', ((upper body)), ((best quality)), ((solo)), ((masterpiece)), (best illustration), ((cinematic light)), hyper detail, (extremely detailed CG unity 8k wallpaper, masterpiece, ultra-detailed)'
        negative_prompt = '((hands)), (((mutation))), (((deformed))), ((ugly)), (worst quality, low quality:1.4),((blurry)), ((horns)), bright, (monochrome:1.3), (oversaturated:1.3), bad hands, lowres, 3d render, cartoon, long body, ((blurry)), duplicate, ((duplicate body parts)), ugly, poor quality, low quality, out of frame'
    #pastel
    if model_dic == 'pastelboys':
        prompt = prompt + f', ((closed shot)), ((upper body)), ((masterpiece)), best quality, ((solo)), (handsome), highly detailed face, colorful Fantastic light and shadow, Scenery, mature,'
        negative_prompt = '((hands)), (worst quality, low quality:1.4), multiple views, 3d render, cartoon, ((blurry)), (text, signature, artist name, watermark), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (too many fingers), (((long neck))), (((mutation))), (((deformed))), ((ugly))'
    #dreamshaper
    if model_dic == 'dreamshaper':
        prompt = prompt + f', ((closed shot)), ((upper body)), ((solo)), (masterpiece), (extremely intricate:1.3), (realistic), detailed, dramatic, award winning, matte drawing, cinematic lighting, octane render, unreal engine'
        negative_prompt = '((hands)), (worst quality, low quality:1.4), canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render'
    #meinamix
    if model_dic == 'meinamix' :
        prompt = prompt + f', ((closed shot)), ((upper body)), ((solo)), (masterpiece), glowing, sidelighting, wallpaper, no hands, wearing clothes'
        negative_prompt = '((hands)), (worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic)'
    #Counterfeit
    if model_dic == 'Counterfeit' :
        prompt = prompt + f', ((closed shot)), ((upper body)), ((masterpiece)), best quality, ((solo)), ((masterpiece,best quality)), blurry background, outdoors'
        negative_prompt = '((hands)), (worst quality, low quality:1.4), multiple views, 3d render, cartoon, ((blurry)), (text, signature, artist name, watermark), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (too many fingers), (((long neck))), (((mutation))), (((deformed))), ((ugly))'
    
    num_samples = 1
    guidance_scale = 8
    num_inference_steps = 50 
    height = 512 
    width = 512 
    
    url = './diffusers_image_output/'+user_id+'/'+url
    init_image = Image.open(url).convert("RGB")
    init_image = init_image.resize((512, 512))  
    
    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            image=init_image,
            strength=0.75,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed)
        ).images
    
    
    file_path = img_rename.make_dic(user_id)
    
    for img in images:
        img.save(file_path + f'/img2img{index}.{seed}.png','PNG')
        with open(file_path + f'/img2img{index}.{seed}.png', 'rb') as one:
                    image_data = one.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        json_object[f'image{index}'] = {
                "image" : encoded_image,
                "prompt" : prompt,
                "seed" : seed,
                "model" : model_dic
            }
        
    return json_object

def main_process(prompt, model_dic,user_id, index, json_object, gpu, url, seed):
    #model_dic : pastelboys or revAnimate
    
    pipe = setting(model_dic, gpu)
    json_object= img2img(model_dic,pipe, prompt, user_id, index, json_object, url, seed)
    return json_object

