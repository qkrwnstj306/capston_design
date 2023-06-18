import gc
import torch
import subprocess
import multiprocessing

def command_txt2img(prompt,model,user_id):
    gc.collect()
    torch.cuda.empty_cache()
    
    proc = subprocess.Popen('bash /home/qkrwnstj/capston_design/invokeai/invoke.sh',stdin= subprocess.PIPE, 
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    
    proc.stdin.write(f'!switch {model}\n')
    proc.stdin.flush()
    
    if model == 'revAnimated_v11':
        prompt = prompt + ', ((upper body)), ((best quality)), ((masterpiece)), (detailed), (best illustration), ((cinematic light)), colorful, hyper detail, dramatic light, intricate details, (extremely detailed CG unity 8k wallpaper, masterpiece, ultra-detailed, best shadow), art by PeterMohrBacher,'+\
            '[NG_DeepNegative_V1_T75, painting by bad-artist, bad_prompt_version2, ((horns)), bright, (monochrome:1.3), (oversaturated:1.3), bad hands, lowres, 3d render, cartoon, long body, ((blurry)), duplicate, ((duplicate body parts)), ugly, poor quality, low quality, out of frame]'
        
    else : 
        prompt = prompt + ', ((masterpiece)), best quality, the most beautiful eyes, (handsome), very short hair, highly detailed face, upper body, magic effect, colorful, fantasy, Fantastic light and shadow, Scenery, mature,'+\
        '[NG_DeepNegative_V1_T75, painting by bad-artist, bad_prompt_version2, (worst quality, low quality:1.4), multiple views, 3d render, cartoon, ((blurry)), (text, signature, artist name, watermark), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (too many fingers), (((long neck))), (((mutation))), (((deformed))), ((ugly))]'
        
    prompt = prompt + f' -n 2 -s 50 -A k_dpmpp_2 -W 512 -H 910 --outdir /home/qkrwnstj/capston_design/invokeai/outputs/{user_id}'
    
    #print('prompt : ',prompt)
    
    #other component
    #-G 0.6 : restore face
    #-U 2 : upscaling
    #-hires_fix   

    

    out, err = proc.communicate(input=str(prompt))
 
 
def main_process(prompt,user_id):
    process1 = multiprocessing.Process(target=command_txt2img, args=(prompt,'revAnimated_v11',user_id))
    process2 = multiprocessing.Process(target=command_txt2img, args=(prompt,'pastelboys2D_v20',user_id))
    process1.start()
    process2.start()
    
    procs = []
    procs.append(process1)
    procs.append(process2)

    for p in procs:
        p.join()  # 프로세스가 모두 종료될 때까지 대기
