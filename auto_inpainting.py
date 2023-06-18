from IPython.display import Image
import os
import numpy as np
import cv2
import subprocess
import torch
from torch import autocast
from diffusers import StableDiffusionInpaintPipelineLegacy, StableDiffusionInpaintPipeline,StableDiffusionImg2ImgPipeline
from PIL import Image
#image name, prompt, negative prompt, seed 값 정해야 됨.
def auto_masking(image_name):
    
    #val_img_path = './example_image/'+prompt+'.png'
    val_img_path = './diffusers_yolo_output_image/'+image_name+'.png'
    #gpu 실행
    #--device 0 


    subprocess.call(f"python ./yolov5/detect.py --weights ./yolo_weight/best.pt --device cpu --img 512 --conf 0.3 --source {val_img_path} \
        --save-txt --name ~/capston_design/yolo_output --exist-ok", shell=True)

    #Image(os.path.join('/home/qkrwnstj/anaconda3/envs/stable_diffusion/capston_design/yolo_output',os.path.basename(val_img_path)))

    #~/anaconda3/envs/stable_diffusion/capston_design/yolo_output # ex1.png, labels -> ex1.txt 로 저장

    #label, ceter_x, ceter_y, width, height
    #img [ 수직 , 수평 ]

    #x_len = int(0.3125*512)
    #y_len = int(0.228516*512)


    img_path = '/home/qkrwnstj/capston_design/yolo_output'
    img = cv2.imread(img_path+'/'+image_name+'.png')

    with open(img_path+'/labels/'+image_name+'.txt','r') as f:
        data = f.readlines()

    data_list = data[0].split(' ')
    data_list[4] = data_list[4][:-2]
    x_len = int(float(data_list[3])*np.shape(img)[1])
    y_len = int(float(data_list[4])*np.shape(img)[0])
    x = int(float(data_list[1])*np.shape(img)[1] - x_len/2)
    y = int(float(data_list[2])*np.shape(img)[0] - y_len/2)

    x2 = int(x+x_len)
    y2 = int(y+y_len)
    print(f'width : {x} ~ {x2}\nheight : {y} ~ {y2}')

    img[:,:] = [0,0,0]
    # masking 영역 넓히는 것도 고려
    img[y:y2,x:x2] = [255,255,255]

    mask_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite("./diffusers_yolo_output_image/"+image_name+"_mask_image.png",mask_image)
    
    #중심좌표, width, height
    return (int((x+x2)/2),int((y+y2)/2)), x_len, y_len

def cal_location(location, x_len, y_len):
    x, y = location # tuple
    x, y = 2*x, 2*y # 1024니까 2배
    x1, y1, x2, y2 = 0,0,0,0
 
    if 1024 - x <= 256:
        x2 = 1024
        x1 = 1024-512
        
    elif x <= 256:
        x1 = 0
        x2 = 512
    
    else :
        x1 = x- 256
        x2 = x+256
        
    if 1024 - y <= 256:
        y2 = 1024
        y1 = 1024-512        
    elif y <= 256:
        y1 = 0
        y2 = 512
    
    else :
        y1 = y-256
        y2 = y+256
    #print(x1,y1,x2,y2)
    return x1,y1,x2,y2
    
    
def auto_inpainting(prompt, negetive, seed, location, img_name, x_len, y_len):
    #location -> 512x512의 중심 x, y 좌표
    
    #hugginface login
    HUGGINGFACE_TOKEN = "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"
    os.system(f'echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token')
    
    url =  "./diffusers_yolo_output_image/ex3.png" #@param {type:"string"}
    origin_image = Image.open(url).convert("RGB")
    origin_image = origin_image.resize((512,512))
    #xy = (512,0,1024,512)
    #xy = (256,0,768,512)
    init_image = Image.open(url).convert("RGB")
    init_image = init_image.resize((1024, 1024))
    
    x1, y1, x2, y2 = cal_location(location, x_len, y_len)
    
    init_image = init_image.crop((x1,y1,x2,y2))
    
    #mask image setting
    mask_url = "./diffusers_yolo_output_image/ex3_mask_image.png"
    mask_image = Image.open(mask_url).convert("RGB")
    mask_image = mask_image.resize((1024, 1024))
    mask_image = mask_image.crop((x1,y1,x2,y2)) # x1,y1, x2,y2
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "Sanster/anything-4.0-inpainting",
    safety_checker=None,
    torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    
    prompt = "beautiful female, higly detailed face, best quality, ((masterpiece)), looking at the viewer, painting by Kyoto Animation, the most beautiful eyes, mature, colorful, fantasy" #@param {type:"string"}
    negative = "(worst quality, low quality:1.4), ((blurry)), (((mutation))), (((deformed))), ((ugly)), ((blur))"#@param {type:"string"}

    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    with autocast("cuda"), torch.inference_mode():
        image = pipe(prompt=prompt,
                negative_prompt=negative,
                num_images_per_prompt=1,
                num_inference_steps=50,
                guidance_scale=7,
                image=init_image, 
                mask_image=mask_image,
                generator = torch.manual_seed(seed)
                ).images
        
    img1 = image[0].resize((256, 256))
    #xy = xy/2
    # (256,0,512,256) for ex2
    # (128,0,384,256) for ex3
    origin_image.paste(img1, (int(x1/2),int(y1/2),int(x2/2),int(y2/2)))  # xy에 자른 이미지를 병합한다. 
    origin_image.save('edit_image.png','PNG')
    
if __name__=='__main__':
    img_name = 'ex3'
    # yolo face detection -> masking -> 중심 좌표, width, height return
    location, x_len, y_len = auto_masking(img_name)
    
    # prompt, negative, seed, img_name 받아서 여기 넣어줘야 됨
    auto_inpainting('test','test',1191447504, location, img_name, x_len, y_len)