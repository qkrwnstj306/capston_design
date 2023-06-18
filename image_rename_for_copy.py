import os

def make_dic(user_id):
    file_path = f'./dreambooth_model_weight/instance image/{user_id}'
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    return file_path
    