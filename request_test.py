import requests
# img2img txt2img copy_image inference
train_path = 'copy_image'
inference_path = 'inference'
sample_path = 'txt2img'
path = sample_path

if path==train_path:
    url = f"https://5f58-210-94-179-16.ngrok-free.app/{train_path}"
    prompt = "(((1girl))), blue eyes, blond hair, short hair"
    dic_name = 'rev_test'
    model_name = 'revAnimate'
    res = requests.post(url, json={"prompt": prompt, "user_id": dic_name, "seed" : "12322", "model" : model_name})

elif path==inference_path:
    url = f"https://5f58-210-94-179-16.ngrok-free.app/{inference_path}"
    instance = 'fDc' # rev : kRd pastel : fDc
    prompt = '((upper body)), wearing suit, in the office, hands in pockets'
    dic_name = 'inference_test_pastel' #rev pastel
    model_name = 'pastelboys' # pastelboys revAnimate
    res = requests.post(url, json={"prompt": prompt, "user_id": dic_name,
                                "instance_prompt" : f"{instance} person", 'model' : model_name})
    
elif path == sample_path:
    url = f"https://5f58-210-94-179-16.ngrok-free.app/{sample_path}"
    prompt = 'a wooded city in summer' # in the street mountain
    dic_name = 'paper_with_kko' #rev pastel
    res = requests.post(url, json={"prompt": prompt, "user_id": dic_name})