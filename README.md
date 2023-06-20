# capston_design
for illustration in web novel

:smile:   
- [ ] add report 
- [ ] add model description   
## code description

**flask_web.py**

1. web server 구동
2. multithreading 기능 (다수의 user 고려)   
3. 경로에 따른 기능 제공   
  - using diffusers
    - /inference 
    - /copy_image
    - /txt2img
    - /img2img
    
  - using invoke AI generation toolkit (application)
    - /16_9
    - /up
    - /user
  


> model의 경우 .ckpt file -> diffusers format으로 바꿔줘야 diffusers library로 불러올 수 있다. 관련 내용은 requirements.txt에도 들어가있음
