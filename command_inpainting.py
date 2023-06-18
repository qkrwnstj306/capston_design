import gc
import torch
import subprocess

def command_inpainting(image_name, prompt, model = 0):
    gc.collect()
    torch.cuda.empty_cache()
    
    original_image_path = "/home/qkrwnstj/capston_design/invokeai/outputs/"+image_name
    mask_image_path = "/home/qkrwnstj/capston_design/yolo_output/"+image_name[:-4]+"_mask_image.png"
    with open(mask_image_path,'rb'):
        new_prompt = '(finely detailed beautiful eyes and detailed face), [(((blur))), ((blurry)), worst face]'
        
        if model == 0: # Rev model
            prompt = prompt + ', ((upper body)), ((best quality)), ((masterpiece)), (detailed), (best illustration), ((cinematic light)), colorful, hyper detail, dramatic light, intricate details, (extremely detailed CG unity 8k wallpaper, masterpiece, ultra-detailed, best shadow), art by PeterMohrBacher,'+\
                '[NG_DeepNegative_V1_T75, painting by bad-artist, bad_prompt_version2, ((horns)), bright, (monochrome:1.3), (oversaturated:1.3), bad hands, lowres, 3d render, cartoon, long body, ((blurry)), duplicate, ((duplicate body parts)), ugly, poor quality, low quality, out of frame]'
            
        else:
            prompt + ', ((masterpiece)), best quality, the most beautiful eyes, (handsome), highly detailed face, upper body, magic effect, colorful, fantasy, Fantastic light and shadow, Scenery, mature,'+\
            '[NG_DeepNegative_V1_T75, painting by bad-artist, bad_prompt_version2, (worst quality, low quality:1.4), multiple views, 3d render, cartoon, ((blurry)), (text, signature, artist name, watermark), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (too many fingers), (((long neck))), (((mutation))), (((deformed))), ((ugly))]'
            
        proc = subprocess.Popen('bash /home/qkrwnstj/capston_design/invokeai/invoke.sh',stdin= subprocess.PIPE, 
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE, universal_newlines=True, shell=True)
        
        inpainting_prompt = prompt +f" {new_prompt}, "+f"-I {original_image_path}"+\
    f" -M {mask_image_path} -s 50 -A k_dpmpp_2 --strength 0.24"

        out, err = proc.communicate(input=str(inpainting_prompt))
 
