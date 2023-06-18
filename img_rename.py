import os

def make_dic(user_id):
    file_path = f'./diffusers_image_output/{user_id}'
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    return file_path
    