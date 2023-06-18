from flask import Flask, jsonify, request
import os, time
import torch, gc, subprocess, multiprocessing
import requests, base64
#for invoekai
import rename_image, rename_image_test
import yolo_inpainting, command_inpainting
import command_txt2img, command_upscaling_X2, command_upscaling_16_9

#for diffusers
import cmd_txt2img, cmd_img2img, cmd_copy_image, training, cmd_dreambooth_inference
import json
import copy 

app = Flask(__name__)
"""using diffusers"""
@app.route('/inference', methods=['POST'])
def inference():
    params = request.get_json()
    model_name = params['model']
    user_id = params['user_id']
    prompt = params['prompt']
    instance_prompt = params['instance_prompt']
    
    model_dic = f'./dreambooth_model_weight/model_weight/{model_name}'
    prompt = instance_prompt +' person, '+ prompt
    json_object = dict()
    gpu = 3
    return_dict = cmd_dreambooth_inference.main_process(prompt, model_dic,user_id, json_object, gpu)
    
    return jsonify(return_dict)

#training 시간 기다리지 않고, return 보내기 위해서 thread 사용.
def thread_copy_img(just_delivery, prompt, model_name, user_id, seed, index, gpu =2):
    with open("./copy_prompt.txt","r") as f:
        for line in f:
            copy_prompt = prompt + line.strip()
            cmd_copy_image.main_process(copy_prompt,model_name, user_id, seed, index, gpu)
            index = index + 1
            
    training.training_dreambooth(model_name, user_id, gpu, just_delivery)
    #os.system(f'rm -r ./dreambooth_model_weight/instance image/{user_id}')
    
@app.route('/copy_image', methods=['POST'])
def copy_txt2img():
    params = request.get_json()
    user_id = params['user_id']
    model_name = params['model']
    seed = int(params['seed'])
    prompt = params['prompt']
    index = 0
    
    print('params for copy image : ', params)
    #DB에 저장되어 있는 작품 index와 등장인물 index
    just_delivery = dict()
    just_delivery['novel_id'] = params['novel_id']
    just_delivery['chrt_id'] = params['chrt_id']
    
    
    #txt file 읽어서 parmas['prompt'] + prompt로 붙이기
    gpu = 2
    p = multiprocessing.Process(target = thread_copy_img, \
        args = (just_delivery, prompt,model_name, user_id, seed, index, gpu))
    p.start()
    
    return jsonify({'message' : 'training start!'})

@app.route('/txt2img', methods=['POST'])
def txt2img():
    start = time.time()
    params = request.get_json()
    print('='*50)
    user_id = params['user_id']
    
    #mapping model to gpu device
    model_dic_lst = {'revAnimate' : '0',
                     'pastelboys' : '1',
                     'dreamshaper' : '2',
                     #'meinamix' : '3',
                     'Counterfeit' : '3'}
    
    #multithreading
    index = 1
    procs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    for model_name in model_dic_lst.keys():
        
        p = multiprocessing.Process(target=cmd_txt2img.main_process
                                , args = (params['prompt'],model_name, user_id, index, return_dict, model_dic_lst[model_name]))
        procs.append(p)
        p.start()
        index = index + 1

    for p in procs:
        p.join()  # 프로세스가 모두 종료될 때까지 대기
    
    #os.system(f'rm -r ./diffusers_image_output/{user_id}')
    end = time.time()
    
    print("수행시간: %f 초" % (end - start))
    print('='*50)
    
    return jsonify(return_dict.copy())

@app.route('/img2img', methods=['POST']) #img to img로 복제 이미지 만들생각이였음
def img2img():
    start = time.time()
    params = request.get_json()
    print('='*50)
    print('user promt : ',params['prompt'])
    print('user id : ',params['user_id'])
    user_id = params['user_id']
    
    #mapping model to gpu device
    model_dic_lst = {'revAnimate' : '2',
                     'pastelboys' : '3',
                     'dreamshaper' : '2',
                     #'meinamix' : '3',
                     'Counterfeit' : '3'}
    
    #user id에 해당하는 directory에 가서 image path 가져오기
    file_list = os.listdir(f'./diffusers_image_output/{user_id}')
    
    #multithreading
    index = 1
    procs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    for model_name in model_dic_lst.keys():
        # 동일한 seed값으로 생성
        file = file_list[index-1]
        left_index = file.find('.')
        right_index = file.rfind('.')
        seed = file[left_index+1:right_index].strip()
    
        p = multiprocessing.Process(target=cmd_img2img.main_process
                                , args = (params['prompt'],model_name, user_id, index, return_dict, model_dic_lst[model_name], file_list[index-1],seed))
        procs.append(p)
        p.start()
        index = index + 1

    for p in procs:
        p.join()  # 프로세스가 모두 종료될 때까지 대기
    
    #os.system(f'rm -r ./diffusers_image_output/{user_id}')
    end = time.time()
    
    print("수행시간: %f 초" % (end - start))
    print('='*50)
    
    return jsonify(return_dict.copy())



"""이 아래는 invokeai application을 사용하여 image를 생성하는 part"""
@app.route('/16_9', methods=['POST'])
def invoke_upscaling_16_9():
    start = time.time()
    params = request.get_json()
    print('='*50)
    print(params['prompt'])
    print('user id : ',params['user_id'])
    user_id = params['user_id']
    command_upscaling_16_9.main_process(params['prompt'])
    response_data = rename_image.rename_image(params['prompt'])
    os.system(f'rm -r /home/qkrwnstj/capston_design/invokeai/outputs/{user_id}')
    end = time.time()
    print("수행시간: %f 초" % (end - start))
    
    return jsonify(response_data)

@app.route('/up', methods=['POST'])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
def invoke_upscaling():
    start = time.time()
    params = request.get_json()
    print('='*50)
    print(params['prompt'])
    print('user id : ',params['user_id'])
    user_id = params['user_id']
    command_upscaling_X2.main_process(params['prompt'])
    response_data = rename_image.rename_image(params['prompt'])
    os.system(f'rm -r /home/qkrwnstj/capston_design/invokeai/outputs/{user_id}')
    end = time.time()
    print("수행시간: %f 초" % (end - start))
    print('='*50)
    return jsonify(response_data)

@app.route('/user', methods=['POST'])
def invoke_txt2img():
    start = time.time()
    params = request.get_json()
    # print('='*50)
    # print('user promt : ',params['prompt'])
    # print('user id : ',params['user_id'])
    user_id = params['user_id']
    
    command_txt2img.main_process(params['prompt'],user_id)
    response_data = rename_image.rename_image(params['prompt'],user_id)
    os.system(f'rm -r /home/qkrwnstj/capston_design/invokeai/outputs/{user_id}')
    end = time.time()
    
    print("수행시간: %f 초" % (end - start))
    # print('='*50)
    #===========image inpainting================
    #각각의 n1이 image name
    #원본은 ./invokeai/outputs/prompt.png   | mask는 .yolo_output/prompt_mask_image.png
    # for num in [n1,n2,n3,n4]:
    #     yolo_inpainting.auto_inpainting(num)
    #     if n3 or n4:
    #         command_inpainting.command_inpainting(num, params['prompt'],1) 
    #     else:
    #         command_inpainting.command_inpainting(num, params['prompt']) 
    
    return jsonify(response_data)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    app.run(port = 4950, debug=True)