import os
import subprocess 
import json
import create_name
import torch 
import requests
#드림부스 다운
'''
subprocess.call('wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py',shell=True)

subprocess.call('wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py',shell=True)

subprocess.call('pip install -qq git+https://github.com/ShivamShrirao/diffusers',shell=True)

subprocess.call('pip install -q -U --pre triton',shell=True)

subprocess.call('pip install -q accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers',shell=True)

#####허깅페이스 경로
path='/home/kkko/capston_design/.huggingface'
os.mkdir(path)
'''

def training_dreambooth(model, user_id, gpu, just_delivery):
    #GPU setting
    torch.cuda.set_device(int(gpu))
    
    HUGGINGFACE_TOKEN = "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"
    os.system(f'echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token')

    #base model
    #맨 처음 model weight를 가져오는 경로
    #MODEL_NAME = f"./civit_model_diffusers/{model}" #@param {type:"string"}
    #이후 model weight 가져 오는 경로
    MODEL_NAME = f"./dreambooth_model_weight/model_weight/{model}"
    
    #weight가 저장될 경로
    OUTPUT_DIR = f"./dreambooth_model_weight/model_weight/{model}"
    print(f"[*] Weights will be saved at {OUTPUT_DIR}")
    
    instance_prompt = create_name.add_name()
    # You can also add multiple concepts here. Try tweaking `--max_train_steps` accordingly.
    concepts_list = [
        {
            "instance_prompt":      f"{instance_prompt} person",
            "class_prompt":         "person",
            "instance_data_dir":    f"./dreambooth_model_weight/instance image/{user_id}",
            "class_data_dir":       "./dreambooth_model_weight/class image"
        }
    ]

    # `class_data_dir` contains regularization images

    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)
        
    #vae : stabilityai/sd-vae-ft-mse
    subprocess.call(f'python3 train_dreambooth.py \
    --pretrained_model_name_or_path={MODEL_NAME} \
    --pretrained_vae_name_or_path="./anime_vae/anything" \
    --output_dir={OUTPUT_DIR} \
    --revision="fp16" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --seed=1337 \
    --resolution=512 \
    --train_batch_size=1 \
    --train_text_encoder \
    --mixed_precision="fp16" \
    --use_8bit_adam \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=50 \
    --sample_batch_size=0 \
    --max_train_steps=2000 \
    --save_interval=10000 \
    --concepts_list="./concepts_list.json"',shell=True)

    print("instance prompt : ", instance_prompt)
    
    
    # 학습 다 했으면 web platform에 완료했다고 알려주기
    just_delivery['model'] = model
    just_delivery['instance_prompt'] = instance_prompt
    url = 'https://dev.hnextits.com/PIIS/trainingDone.sa'
    requests.post(url, json=just_delivery)
    
    #return instance_prompt

# if __name__=='__main__':
#     training_dreambooth('revAnimate','rev_test',3)