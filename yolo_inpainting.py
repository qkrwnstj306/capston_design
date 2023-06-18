from IPython.display import Image
import os
import numpy as np
import cv2
import subprocess


def auto_inpainting(image_name):
    prompt = image_name[:-4]
    #val_img_path = './example_image/'+prompt+'.png'
    val_img_path = './invokeai/outputs/'+prompt+'.png'
    #gpu 실행
    #--device 0 


    subprocess.call(f"python ./yolov5/detect.py --weights ./yolo_weight/best.pt --device cpu --img 512 --conf 0.3 --source {val_img_path} --save-txt --name ~/capston_design/yolo_output --exist-ok", shell=True)

    #Image(os.path.join('/home/qkrwnstj/anaconda3/envs/stable_diffusion/capston_design/yolo_output',os.path.basename(val_img_path)))

    #~/anaconda3/envs/stable_diffusion/capston_design/yolo_output # ex1.png, labels -> ex1.txt 로 저장

    #label, ceter_x, ceter_y, width, height
    #img [ 수직 , 수평 ]

    #x_len = int(0.3125*512)
    #y_len = int(0.228516*512)


    img_path = '/home/qkrwnstj/capston_design/yolo_output'
    img = cv2.imread(img_path+'/'+prompt+'.png')

    with open(img_path+'/labels/'+prompt+'.txt','r') as f:
        data = f.readlines()

    print(np.shape(img))
    print(np.shape(img)[0])
    data_list = data[0].split(' ')
    data_list[4] = data_list[4][:-2]
    x_len = int(float(data_list[3])*np.shape(img)[0])
    y_len = int(float(data_list[4])*np.shape(img)[1])
    x = int(float(data_list[1])*np.shape(img)[0] - x_len/2)
    y = int(float(data_list[2])*np.shape(img)[1] - y_len/2)

    x2 = int(x+x_len)
    y2 = int(y+y_len)
    print(f'width : {x} ~ {x2}\nheight : {y} ~ {y2}')

    img[y:y2,x:x2] = [0,0,0]

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, mask_image = cv2.threshold(imgGray, 0.00001,255,cv2.THRESH_BINARY)

    #invoke에서는 반전시킬 필요가 없음
    # for row_index in range(len(mask_image)):
    #     for col_index in range(len(mask_image[row_index])):
    #         if mask_image[row_index][col_index]==255:
    #             mask_image[row_index][col_index] = 0
    #         else :
    #             mask_image[row_index][col_index] = 255

    cv2.imwrite("./yolo_output/"+prompt+"_mask_image.png",mask_image)
    
    #원본은 ./invokeai/outputs/prompt.png   | mask는 .yolo_output/prompt_mask_image.png
    