---
layout: single
title:  "05 face_drowsiness detection"
categories: Until_YOLO
tag: [python, opencv, face detection, drowsiness]
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>



```python
import numpy as np
import dlib
import cv2

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))

predictor_file = 'shape_predictor_68_face_landmarks.dat'
image_file = 'KJY.jpg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)
print("Number of faces detected: {}".format(len(rects)))


for (i, rect) in enumerate(rects):
    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    show_parts = points[EYES]
    for (i, point) in enumerate(show_parts):
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        cv2.putText(image, "{}".format(i + 1), (x, y - 2),
		cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

cv2.imshow("Face Landmark", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
![a](https://user-images.githubusercontent.com/105587839/207239694-2bf30836-25a0-44a6-8486-71cd3f5b2e30.png)

<pre>
Number of faces detected: 1
</pre>


점들의 Euclidean Distance를 활용하여 눈이 떠있는지 판단할 수 있다.


## EAR


![b](https://user-images.githubusercontent.com/105587839/207239713-93dee8ae-6bf7-4cad-a381-61ae455b2012.jpg)



- 각 눈은 눈의 왼쪽 corner에서 시작하여 6(x, y)좌표로 표시된 다음 나머지 영역에서 시계 방향으로 작동

- 이 좌표의 너비와 높이 사이에는 관계가 있습니다.

- Soukupová와 Čech의 Real-Time Eye Blink Detection using Facial Landmarks,2016 논문에서 face landmark를 사용한 실시간 눈 깜박임 감지를 기반으로 eye aspect ratio(EAR)라고하는이 관계를 반영하는 방정식을 도출 할 수 있습니다.

![c](https://user-images.githubusercontent.com/105587839/207239818-0f5866f7-d019-47d5-923c-c371d0520aee.jpg)

   
  
## face_drowsiness detection



```python
import numpy as np
import dlib
import cv2
import time
import pygame  #game을 하기 위한 python, 소리를 내기위해서 사용했음.

pygame.mixer.init()  #소리를 넣어줌
pygame.mixer.music.load('fire-truck.wav')

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))
frame_width = 640
frame_height = 480

title_name = 'Face Drowsiness Detection'
elapsed_time = 0  #측정시간을 0으로 둠

face_cascade_name = 'haarcascade_frontalface_alt.xml'  #haarcascades가 빠르기 때문에 haarcascades 활용함(단점: accuracy가 떨어짐)
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

predictor_file = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_file)

status = 'Awake'
number_closed = 0  #몇 번을 졸았는지
min_EAR = 0.15  #사용해보고 변경
closed_limit = 15  #15번 이상 졸면 졸은 것으로 판단
show_frame = None
sign = None
color = None

def getEAR(points):
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)
    
def detectAndDisplay(image):  #하나 하나 image를 받아옴
    global number_closed
    global color
    global show_frame
    global sign
    global elapsed_time
    start_time = time.time()  #시작하는 시간 입력
    #height,width = image.shape[:2]
    image = cv2.resize(image, (frame_width, frame_height))  #image resize
    show_frame = image
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))
        points = np.matrix([[p.x, p.y] for p in predictor(frame_gray, rect).parts()])
        show_parts = points[EYES]
        right_eye_EAR = getEAR(points[RIGHT_EYE])
        left_eye_EAR = getEAR(points[LEFT_EYE])
        mean_eye_EAR = (right_eye_EAR + left_eye_EAR) / 2 

        right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype("int")
        left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype("int")

        cv2.putText(image, "{:.2f}".format(right_eye_EAR), (right_eye_center[0,0], right_eye_center[0,1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, "{:.2f}".format(left_eye_EAR), (left_eye_center[0,0], left_eye_center[0,1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for (i, point) in enumerate(show_parts): #눈에 점을 찍어줌
            x = point[0,0]
            y = point[0,1]
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
            
        if mean_eye_EAR > min_EAR:
            color = (0, 255, 0)
            status = 'Awake'
            number_closed = number_closed - 1
            if( number_closed<0 ):
                number_closed = 0
        else:
            color = (0, 0, 255)
            status = 'Sleep'
            number_closed = number_closed + 1
                     
        sign = status + ', Sleep count : ' + str(number_closed) + ' / ' + str(closed_limit)
        if( number_closed > closed_limit ):
            show_frame = frame_gray
            # play SOUND
            if(pygame.mixer.music.get_busy()==False):
                pygame.mixer.music.play()

    cv2.putText(show_frame, sign , (10,frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow(title_name, show_frame)
    frame_time = time.time() - start_time
    elapsed_time += frame_time
    print("Frame time {:.3f} seconds".format(frame_time))
    

vs = cv2.VideoCapture(0)
time.sleep(2.0)  #카메라 켜지는거 살짝 기다려줌
if not vs.isOpened:
    print('### Error opening video ###')
    exit(0)
while True:
    ret, frame = vs.read()
    if frame is None:
        print('### No more frame ###')
        vs.release()
        break
    detectAndDisplay(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vs.release()
cv2.destroyAllWindows()
```

<pre>
Frame time 0.072 seconds
Frame time 0.054 seconds
Frame time 0.057 seconds
Frame time 0.065 seconds
Frame time 0.053 seconds
Frame time 0.048 seconds
Frame time 0.050 seconds
Frame time 0.061 seconds
Frame time 0.054 seconds
Frame time 0.053 seconds
Frame time 0.051 seconds
Frame time 0.057 seconds
Frame time 0.051 seconds
Frame time 0.050 seconds
Frame time 0.050 seconds
Frame time 0.055 seconds
Frame time 0.051 seconds
Frame time 0.052 seconds
Frame time 0.049 seconds
Frame time 0.056 seconds
Frame time 0.049 seconds
Frame time 0.049 seconds
Frame time 0.052 seconds
Frame time 0.060 seconds
Frame time 0.053 seconds
Frame time 0.051 seconds
Frame time 0.050 seconds
Frame time 0.053 seconds
Frame time 0.049 seconds
Frame time 0.050 seconds
Frame time 0.053 seconds
Frame time 0.065 seconds
Frame time 0.051 seconds
Frame time 0.051 seconds
Frame time 0.054 seconds
Frame time 0.062 seconds
Frame time 0.053 seconds
Frame time 0.053 seconds
Frame time 0.051 seconds
Frame time 0.052 seconds
Frame time 0.051 seconds
Frame time 0.051 seconds
Frame time 0.051 seconds
Frame time 0.052 seconds
Frame time 0.052 seconds
Frame time 0.049 seconds
</pre>
![d](https://user-images.githubusercontent.com/105587839/207239770-64e91d18-6e0d-47d7-87c2-b78415c21906.png)

