import string, random
# (1) 검사가 완벽하지 않음 (재검사를 안함)
# (2) 대문자 소문자 구별해서 나옴 경우의 수는 많지만 효율적이지 않음
def create_name():
    rand1 = random.choice(string.ascii_letters)
    rand2 = random.choice(string.ascii_letters)
    rand3 = random.choice(string.ascii_letters)
    rand=rand1+rand2+rand3    
    return rand
def add_name():    
    rand = create_name()    
    with open('name_list.txt','r') as f:
        name_lst = f.readlines()
        for name in name_lst:
            if rand == name.strip():
                print("겹칩니다")
                rand = create_name()   
    
    with open('name_list.txt','a') as f:
        f.write('\n'+rand)
        f.close()
        
    return rand