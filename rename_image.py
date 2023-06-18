 # 이미지 파일을 바이트 형태로 읽어온 후, base64로 인코딩하여 JSON으로 응답합니다.
import base64
import os

# 요청한 user에게 정확히 image 전달
def rename_image(prompt,user_id):
    dir_path = f"/home/qkrwnstj/capston_design/invokeai/outputs/{user_id}"
   
    file_list = os.listdir(dir_path)
    response_data = {}
    index = 0
    
    for file in file_list:
        if file[-3:]=='png': # .txt 제외하고 .png만 전달
            with open(dir_path+'/'+file, 'rb') as one:
                    #print(f'image{index} : ', file)
                    left_index = file.find('.')
                    right_index = file.rfind('.')
                    seed = file[left_index+1:right_index].strip()
                    #print(f'seed{index} : ',seed)
                    image_data = one.read()
                    encoded_image = base64.b64encode(image_data).decode('utf-8')
                    response_data[f'image'+str(index)] = encoded_image
                    index += 1
        
    return response_data
   