# capston_design
for illustration in web novel

:smile:   
- [ ] add report 
- [ ] add model description   

## Goal

본 프로젝트를 통해, 작가가 삽화가에게 삽화 외주를 맡기는 과정에서 발생하는 문제를 해결하고자 한다.   
웹 소설 플랫폼에 자체적인 AI model을 구축함으로써, 안정적인 system을 유지한다.   
발생하는 효과들
---
*시간적, 금전적 문제 해결*
*작가가 원하는 등장인물 생성*
*Tag를 통한 간단하고 편리한 생성*
*작가의 의견 즉각 반영*
*30초 정도의 짧은 시간으로 이미지 생성*

## code description

**flask_web.py**

1. web server 구동
2. multithreading 기능 (다수의 user 고려)   
3. 경로에 따른 기능 제공   
  - using diffusers
    - /txt2img
    
    - /copy_image
    - /inference
    - /img2img
    
  - using invoke AI generation toolkit (application)
    - /16_9
    - /up
    - /user
  


> model의 경우 .ckpt file -> diffusers format으로 바꿔줘야 diffusers library로 불러올 수 있다. 관련 내용은 requirements.txt에도 들어가있음
