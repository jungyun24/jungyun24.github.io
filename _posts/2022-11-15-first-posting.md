---
layout: single
title: "Lecture 1 : Introduction"
categories: CS231n
tags: [stanford, lecture, review, CS231n]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---
# Lecture 1 | Introduction to Convolutional Neural Networks for Visual Recognition
<iframe width="560" height="315" src="https://www.youtube.com/embed/vT1JzLTH4G4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

***
   
## What is Computer Vision? 

![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-1](https://user-images.githubusercontent.com/105587839/201809020-b1673121-0c50-4a9c-b695-13203293c5c4.jpg)
   
CS231n은 Computer Vision에 관한 것입니다.   
Computer Vision은 visual data에 대한 연구입니다.   
   
전세계적으로 sensor가 많아 visual data가 엄청나게 많은 양이 쏟아지고 있다.   
이 data를 활용하고 이해할 수 있는 algorithm을 개발하는 것이 매우 중요하다.   
Computer Vision은 문제가 있는데 우주의 암흑물질처럼 인터넷의 visual data 또한 algorithm이 실제로 web의 모든 visual data를 구성하는 것이 정확이 무엇인지 이해하고 확인하는 것이 매우 어렵다.(ex. youtube)   
   
![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-2](https://user-images.githubusercontent.com/105587839/201809294-a41ce56f-27c8-41a8-861c-94208aaf2599.jpg)
   
Computer Vision은 많은 다양한 분야의 science, engineering 그리고 technology을 다루고 있다.  

***
   
## Today's agenda

![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-5](https://user-images.githubusercontent.com/105587839/201809315-faacc3e6-c65b-4b96-a4a1-7439d646d248.jpg)
    
간단한 Computer Vision의 역사와 CS231n의 overview를 할 것이다.  

***

### A brief history of Computer Vision

![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-6](https://user-images.githubusercontent.com/105587839/201809323-4ac84d8e-be69-4b4a-b078-0afeafd1fcf2.jpg)
   
(Biological Vision)   
Vision의 역사는 약 5억 4300만년 전으로 거슬러 올라갑니다.   
지구는 대부분 물이었고 동물은 많이 움직이지도 않고 눈도 없고 떠다녔습니다.   
약 5억 4천만년 전 놀라운 일이 일어납니다.   
   
천만년 안에 동물 종의 수가 폭발적으로 증가했다는 사실을 발견합니다.   
원인은 무엇일까요?   
많은 이론이 있었지만 Andrew Parker가 가장 설득력 있는 이론 중 하나를 제안합니다.   
최초의 동물은 눈을 발달시켰고 vision의 시작은 폭발적인 종분화 단계를 시작했다는 것입니다.   
이로 인해 삶이 훨씬 더 능동적이 되었습니다.   
예를 들면, 일부 predators는 먹이를 쫓았고 먹이는 predators로부터 도망을 다녔습니다.   
   
동물은 생존하기 위해 빠르게 진화해야 했기 때문에 시각은 거의 모든 동물의 가장 큰 감각 system으로 발전했습니다.   
특히, 지능이 있는 인간은 visual processing에 관여하는 피질 neuron의 거의 50%를 가지고 있습니다.   
   
![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-7](https://user-images.githubusercontent.com/105587839/201809328-5c83d22a-8794-477a-92eb-4696008d0ca1.jpg)
   
(Mechanical Vision)   
오늘 날 우리가 알고 있는 초기 camera 중 하나는 1600년대 르네상스 시대의 Camera Obscura이며,   
이는 pinhole camera 이론에 기반한 camera입니다.   
빛을 수집하는 구멍을 통해 정보를 수집하고 image를 투사하는 camera 뒷면의 평면으로 발달한 동물의 초기 눈과 매우 유사합니다.   
   
![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-8](https://user-images.githubusercontent.com/105587839/201809341-4a4ea6b9-d054-43bc-ad14-bff8254a0748.jpg)
   
Biologists는 vision의 mechanisem을 연구하기 시작했습니다.   
인간의 vision에서 가장 영향력 있는 작업 중 하나는 아래와 같습니다.   
동물의 vision과 inspired된 computer vision이 electrophysiology를 사용하여 50년대와 60년대에 Hubel과 Wiesel이 수행한 작업입니다.   
"primates(영장류)와 mammals(포유류)의 visual processing mechanism은 무엇인가?"   
따라서, 인간의 뇌와 유사한 고양이의 뇌로 연구를 하였습니다.   
   
the primary visual cortex(일차 시각 피질) 영역이 있는 고양이 뇌의 뒤쪽에 전극을 몇개 붙인 다음 어떤 자극이 고양이 뇌의 the primary visual cortex(일차 시각 피질) 뒤쪽에 잇는 neuron이 흥분하게 반응하는지 살펴보는것입니다.   
결론은, 고양이 뇌의 the primary visual cortex(일차 시각 피질) 부분에 많은 유형의 세포가 있다는 것입니다.   
가장 중요한 세포 중 하나는 그들이 특정 방향으로 움직일 때 oriented edges에 반응하는 단순한 세포입니다.   
그들이 발견한 것은, visual processing가 visual world의 단순한 구조, oriented edges에서 시작하고 정보가 visual processing 경로를 따라 이동함에 따라 뇌가 visual information의 복잡성을 축적한다는 것입니다.   
   
![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-9](https://user-images.githubusercontent.com/105587839/201809352-d7e190db-523a-46e6-bfc3-89ad8161d960.jpg)
   
Computer Vision의 역사도 60년대 초에 시작됩니다.   
Block World는 Larry Roberts가 출판한 일련의 작품으로 아마도 computer vision의 첫번째 PhD 논문일 것입니다.
visual world가 simple한 geometric shapes으로 simplified하였고, 이 shapes을 무엇인지 recognize하고 이런 shapes을 reconstruction하는것이 목표였습니다.   
   
![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-10](https://user-images.githubusercontent.com/105587839/201809360-2e2f078a-3eb9-44d1-9635-aa2e587cd21a.jpg)
   
1966년에 "The Summer Vision Project"라는 유명한 MIT 여름 프로젝트가 있었습니다.   
"The summer Vision Project"의 목표는 "우리의 summer workers가 효과적으로 visual system의 중요한 부분을 construction을 시도하는 것입니다."   
(여름동안 visual system의 대부분을 해결하겠다!!)   
이후 50년이 흘렀고 computer vision 분야는 한 여름 project에서 꽃을 피웠고 전세계적으로 수천 명의 연구자들이 vision의 가장 근본적인 문제 중 일부를 연구하고 있습니다.   
우리는 아직 vision을 풀지 못했지만 인공지능 분야에서 가장 중요하고 빠르게 성장하는 분야 중 하나로 성장했습니다.   
   
![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-11](https://user-images.githubusercontent.com/105587839/201809369-9a3b0ad6-cd70-4b71-81c2-ff68f150c5be.jpg)
   
David Marr는 MIT에서 vision 과학자였으며, 70년 대 후반에 자신이 생각하는 vision이 무엇인지, computer vision과 computer가 visual world를 recognize할 수 있는 algorithm을 개발하는 방법에 대해 영향력 있는 책을 저술했습니다.   
   

![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-12](https://user-images.githubusercontent.com/105587839/201809371-2cd4695b-1b1a-42e3-aae1-0b3b5bd5efb5.jpg)
   
David Marr의 저서에서 생각하는 과정은 image를 찍고 최종적으로 holistic full 3d representation(전체론적 전체 3D표현)에 도달하기 위해서는 몇 가지 과정을 거쳐야 한다는 것입니다.   
   
1st Process : Primal Sketch
 - Primal Sketch는 대부분 the edges, the bars, the ends, the virtual lines, the curves, the boundaries를 represented되는 곳입니다.
 - neuroscientists(신경과학자)들이 본 것 (Hubel과 Wiesel은 visual processing의 초기 단계는 edge와같은 simple structure와 관련이 있다는것)에서 많은 영감을 받았습니다.
   
2nd Process : 2.5D Sketch & 3rd Process : 3D Model Representation
- the surfaces, the depth information, the layers 또는 the discontinuities of the visual scene을 조합하기 시작한 다음 결국 모든 것을 조합하고 surface 및 volumetric primitives(체적 기본요소) 등의 측면에서 hierarchically(계층적으로) 구성된 3D model을 갖게 됩니다.   
   
그래서 vision이 무엇인지에 대한 매우 이상적인 사고 과정이었고 이러한 사고 방식은 실제로 수십 년 동안 computer vision을 지배해 왔습니다.   
   
![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-13](https://user-images.githubusercontent.com/105587839/201809376-92405d4f-126e-4911-81a8-4a20c7a02432.jpg)
   
70년대, 또 다른 매우 중요한 작업 group은 question을 묻기 시작했습니다.   
"우리가 어떻게 단순한 block world를 넘어 real world objects를 recognizing하거나 representing을 시작할 수 있을까?"   
70년대는 사용 가능한 data가 거의 없고 느리고 pc는 대중화되지 않았습니다.   
   
우리가 object를 recognize하고 represent할 수 있는 방법은   
Stanford와 SRI의 Palo Alto에서 유사한 아이디어를 제안하는 group이 있습니다.     
하나는 " Generalized cylinder", 다른 하는 "Pictorial Struture"   
기본 아이디어는 모든 object가 simple geometric primitives로 구성된다는 것입니다.   
(Ex. 사람은 generalized된 원통 모양으로 연결 혹은 이러한 부분 사이의 elastic distance에서 중요한 부분으로 연결될 수 있습니다.)   
    
![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-14](https://user-images.githubusercontent.com/105587839/201809384-ee567c3c-2e72-4270-96e4-fb684e5a77ad.jpg)
(영상에서는 면도기이지만 결국 같은 의미입니다)
   
80년대, David Lowe는 simple world structures에서 visual world를 reconstruct하거나 recognize하는 방법을 생각하는 또른 예입니다.   
lines과 edges, 그리고 대부분 straight lines와 그들의 조합을 constructing하여 동전(영상에서는 면도기)을 recognize합니다.   
   
   
   
**60, 70, 80년대에 computer vision의 과제가 무엇인지 생각해보려는 노력이 많았고 object recognition 문제를 해결하는 것은 매우 어려웠습니다.**
   
   
   
   
![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-15](https://user-images.githubusercontent.com/105587839/201809389-e1f25858-3be8-485b-8bd6-b552175ba9ef.jpg)
   
object recognition이 너무 어렵다면 먼저 object segmentation을 수행하자!!   
즉, image를 가져와 pixels을 의미 있는 영역으로 그룹화하는 작업입니다.   
(Ex. 같이 그룹화되는 pixels을 사람이라고 하는지는 모를 수 있지만, 배경에서 사람에 속하는 모든 pixels을 추출할 수 있습니다.)   
   
Jitendra Malik와 Berkeley의 그의 학생 Jianbo Shi가 image segmentation에 대한 graph theory algorithm을 사용한 아주 초기의 중요한 작업이 있습니다.(ppt그림)   
   
![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-16](https://user-images.githubusercontent.com/105587839/201809403-7c2a827c-c3d0-424a-900e-b81b965df712.jpg)
   



![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-17](https://user-images.githubusercontent.com/105587839/201809407-bbcc3978-65e9-4e8f-bbd9-66440c8fbf6b.jpg)



![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-18](https://user-images.githubusercontent.com/105587839/201809412-afa3b332-3810-47a4-91d9-690aac092683.jpg)



![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-19](https://user-images.githubusercontent.com/105587839/201809423-87b6e651-b920-4efe-aa96-6856e7043346.jpg)



![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-20](https://user-images.githubusercontent.com/105587839/201809430-80f53ac4-d890-474c-9093-edf278540001.jpg)



![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-21](https://user-images.githubusercontent.com/105587839/201809436-a1246e56-475c-4eec-94a3-c38354a08315.jpg)



![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-22](https://user-images.githubusercontent.com/105587839/201809444-1e943bb4-bc53-44a7-92c5-fe67b94576c1.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-23](https://user-images.githubusercontent.com/105587839/201809448-14e1cb6d-a091-4176-8d0e-a72a45131e17.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-24](https://user-images.githubusercontent.com/105587839/201809457-5906217b-441a-434f-9551-7e379912111d.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-25](https://user-images.githubusercontent.com/105587839/201809465-028b81a0-7dd1-476e-909c-6706bb368814.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-26](https://user-images.githubusercontent.com/105587839/201809476-362a5914-9ecc-4bc9-b70e-092dc81aa91b.jpg)

![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-27](https://user-images.githubusercontent.com/105587839/201809481-c7318103-ecf5-430b-b513-6151199b930e.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-28](https://user-images.githubusercontent.com/105587839/201809486-7dbc8f73-e7de-4d98-9f40-6a505a603a8d.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-29](https://user-images.githubusercontent.com/105587839/201809494-f750ada4-1e41-46f9-8652-6bc519433f65.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-30](https://user-images.githubusercontent.com/105587839/201809497-236b966e-7e52-433b-ade1-9df23c3935a6.jpg)

![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-31](https://user-images.githubusercontent.com/105587839/201809545-8c7d8dd4-2ed9-4d99-b59d-e91afc7b1734.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-32](https://user-images.githubusercontent.com/105587839/201809557-ffbbe27c-4601-410e-8583-f728b23f6afe.jpg)




![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-33](https://user-images.githubusercontent.com/105587839/201809561-e6abfa85-d0da-4b7c-a66f-8aa4b4fc278e.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-34](https://user-images.githubusercontent.com/105587839/201809563-ad1c2845-de54-4c17-9a23-157399440a1b.jpg)




![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-35](https://user-images.githubusercontent.com/105587839/201809567-19c5b989-28f1-47ce-86ab-0e292cc26dc7.jpg)

![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-36](https://user-images.githubusercontent.com/105587839/201809572-b0d79902-c62f-40e3-9e8d-5a75250d1ab5.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-37](https://user-images.githubusercontent.com/105587839/201809577-653e362f-e9db-4736-8f59-cbea44e4f878.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-38](https://user-images.githubusercontent.com/105587839/201809580-19ae2340-45ef-41c9-ada1-85d1998eddad.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-39](https://user-images.githubusercontent.com/105587839/201809586-21f82a44-3710-4a66-8c47-3871af01d476.jpg)



![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-40](https://user-images.githubusercontent.com/105587839/201809602-760c9472-06e2-464f-b4c3-7f7405f617a9.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-41](https://user-images.githubusercontent.com/105587839/201809610-c4a0d617-1209-4d91-b2a7-28b562027716.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-42](https://user-images.githubusercontent.com/105587839/201809615-e6a9815b-8114-4440-93fe-893884cb6081.jpg)



![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-43](https://user-images.githubusercontent.com/105587839/201809621-e0380ca3-9ef1-407e-af06-30de0e80a451.jpg)



![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-44](https://user-images.githubusercontent.com/105587839/201809624-f7038d48-9d42-4d9d-a62f-b7996fa02893.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-45](https://user-images.githubusercontent.com/105587839/201809627-34171327-7862-477a-ac8a-0c57c06bbc40.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-46](https://user-images.githubusercontent.com/105587839/201809629-fe1e5cbb-746f-4adb-828f-b569650e24e3.jpg)


![bfece0e701ea4502eefa396d166da777rpczjE82RwKejnPq-47](https://user-images.githubusercontent.com/105587839/201809633-a67ec075-ea23-4da3-b00f-a05c0aedf093.jpg)



































































