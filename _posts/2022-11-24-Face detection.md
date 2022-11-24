---
layout: single
title:  "04 Face Detection"
categories: Until_YOLO
tag: [coding, opencv, convert, image, python, computervision, face detection]
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


# Face Detection


1. Haar cascade

    - 굉장히 가벼움 (라즈베리 파이와 같이 제한된 환경에선 good)

    - accuracy가 떨어짐

  

  

2. Deep learning

    - OpenCV에 dnn module이 들어가면서 생긴 방식

    - 말그대로 deep learning을 이용한 방식

    - 좀 더 정확, 다양한 기능 제공

    - Haar보다는 무거움

  

  

3. face_recognition

    - library를 이용한 방식

    - accuracy 괜찮음



# Face Landmark



```python
import numpy as np
import dlib   # c++기반 library, landmark를 recognition하는데 사용
import cv2

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))

predictor_file = 'shape_predictor_68_face_landmarks.dat'  #training된 model(68개 점을 찍어줌)
image_file = 'graduate.png'

detector = dlib.get_frontal_face_detector()   #정면 사진을 detection하겠다
predictor = dlib.shape_predictor(predictor_file)  #68개 점을 가져옴

image = cv2.imread(image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #noise를 줄여주려고 filter 사용

rects = detector(gray, 1)   #detection을 할때 image layer를 어떻게 할까? 1은 큰 image
print("Number of faces detected: {}".format(len(rects)))  # rect에 얼굴 몇 개 recognition했나 보여줌


for (i, rect) in enumerate(rects):
    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    show_parts = points[ALL] #보고싶은 부분 ALL 대신 적으면 됨 ex)MOUTH, EYES...
    for (i, point) in enumerate(show_parts):
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)  #노랑색 점
        cv2.putText(image, "{}".format(i + 1), (x, y - 2), #점 좌표
		cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

cv2.imshow("Face Landmark", image)
cv2.waitKey(0)   
cv2.destroyAllWindows()
```

<pre>
Number of faces detected: 6
</pre>
![01](https://user-images.githubusercontent.com/105587839/203671211-566cac83-134a-42f2-9890-2feb5acf0c32.png)


## Face Alignment


기존의 image의 인식률을 높이기 위해 사용



```python
import numpy as np
import dlib
import cv2

#눈에 해당하는 점만 사용할 것이다.
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

predictor_file = 'shape_predictor_68_face_landmarks.dat'
image_file = 'holiday_jeju.jpg'
MARGIN_RATIO = 1.5  #얼굴을 인식한 것보다 조금 크게 해서 여유 있게 공간을 만듦
OUTPUT_SIZE = (300, 300)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
image_origin = image.copy()  #original에서 보여주려고 하나를 copy

(image_height, image_width) = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1) #한번만 upscale

def getFaceDimension(rect):
    return (rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top())

def getCropDimension(rect, center):
    width = (rect.right() - rect.left())
    half_width = width // 2
    (centerX, centerY) = center
    startX = centerX - half_width
    endX = centerX + half_width
    startY = rect.top()
    endY = rect.bottom() 
    return (startX, endX, startY, endY)    

for (i, rect) in enumerate(rects):  #얼굴 갯수에 따라
    (x, y, w, h) = getFaceDimension(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  #녹색 사각형 그림

    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    show_parts = points[EYES]  #눈만 가져옴

    right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype("int")
    left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype("int")
    print(right_eye_center, left_eye_center)
 
    # eye center에 red point를 찍을 것이다.
    cv2.circle(image, (right_eye_center[0,0], right_eye_center[0,1]), 5, (0, 0, 255), -1)  
    cv2.circle(image, (left_eye_center[0,0], left_eye_center[0,1]), 5, (0, 0, 255), -1)
    
    cv2.circle(image, (left_eye_center[0,0], right_eye_center[0,1]), 5, (0, 255, 0), -1)
    
    # eye의 triangle
    cv2.line(image, (right_eye_center[0,0], right_eye_center[0,1]),
             (left_eye_center[0,0], left_eye_center[0,1]), (0, 255, 0), 2)
    cv2.line(image, (right_eye_center[0,0], right_eye_center[0,1]),
         (left_eye_center[0,0], right_eye_center[0,1]), (0, 255, 0), 1)
    cv2.line(image, (left_eye_center[0,0], right_eye_center[0,1]),
         (left_eye_center[0,0], left_eye_center[0,1]), (0, 255, 0), 1)

    eye_delta_x = right_eye_center[0,0] - left_eye_center[0,0]
    eye_delta_y = right_eye_center[0,1] - left_eye_center[0,1]
    degree = np.degrees(np.arctan2(eye_delta_y,eye_delta_x)) - 180
    
    #피타고라스로 distance를 구함
    eye_distance = np.sqrt((eye_delta_x ** 2) + (eye_delta_y ** 2))
    
    #밑변 distance
    aligned_eye_distance = left_eye_center[0,0] - right_eye_center[0,0]
    scale = aligned_eye_distance / eye_distance

    eyes_center = (int((left_eye_center[0,0] + right_eye_center[0,0]) // 2),
            int((left_eye_center[0,1] + right_eye_center[0,1]) // 2))
    cv2.circle(image, eyes_center, 5, (255, 0, 0), -1)
            
    metrix = cv2.getRotationMatrix2D(eyes_center, degree, scale)
    cv2.putText(image, "{:.5f}".format(degree), (right_eye_center[0,0], right_eye_center[0,1] + 20),
     	 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #original을 돌림
    warped = cv2.warpAffine(image_origin, metrix, (image_width, image_height),
        flags=cv2.INTER_CUBIC)
    
    cv2.imshow("warpAffine", warped)
    (startX, endX, startY, endY) = getCropDimension(rect, eyes_center) #자를 지점
    croped = warped[startY:endY, startX:endX]
    output = cv2.resize(croped, OUTPUT_SIZE)
    cv2.imshow("output", output)

    for (i, point) in enumerate(show_parts): #눈에 해당
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)

cv2.imshow("Face Alignment", image)
cv2.waitKey(0)   
cv2.destroyAllWindows()
```

<pre>
[[333 393]] [[353 390]]
</pre>

![02](https://user-images.githubusercontent.com/105587839/203671202-ad0cdc7d-9e3d-4133-bf93-2fe614cbe569.png)


## Increase recognition


<성능이 낮은 이유>

1. image 수가 적다

2. image에서 정면을 쳐다보지 않는다

3. 해상도와 크기가 작다



```python
import numpy as np
import dlib
import cv2

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

dataset_paths = ['./tedy-front/', './son-front/', './unknown-front/']
output_paths = ['./tedy-align/', './son-align/', './unknown-align/']
number_images = 20
image_type = '.jpg'

predictor_file = 'shape_predictor_68_face_landmarks.dat'
MARGIN_RATIO = 1.5
OUTPUT_SIZE = (300, 300)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

def getFaceDimension(rect):
    return (rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top())

def getCropDimension(rect, center):
    width = (rect.right() - rect.left())
    half_width = width // 2
    (centerX, centerY) = center
    startX = centerX - half_width
    endX = centerX + half_width
    startY = rect.top()
    endY = rect.bottom() 
    return (startX, endX, startY, endY)    

for (i, dataset_path) in enumerate(dataset_paths):  #dataset들을 가져옵니다.
    # 사람 이름을 찾습니다
    output_path = output_paths[i]
    
    for idx in range(number_images):
        input_file = dataset_path + str(idx+1) + image_type

        # get RGB image from BGR, OpenCV format
        image = cv2.imread(input_file)
        image_origin = image.copy()

        (image_height, image_width) = image.shape[:2]  #image 크기 구함
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #Color noise 없애줌

        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            (x, y, w, h) = getFaceDimension(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])  #gray image에서 68개 point를 찾음
            show_parts = points[EYES]  #그 중 눈만 보여줌

            right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype("int")
            left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype("int")

            eye_delta_x = right_eye_center[0,0] - left_eye_center[0,0]
            eye_delta_y = right_eye_center[0,1] - left_eye_center[0,1]
            degree = np.degrees(np.arctan2(eye_delta_y,eye_delta_x)) - 180

            eye_distance = np.sqrt((eye_delta_x ** 2) + (eye_delta_y ** 2))
            aligned_eye_distance = left_eye_center[0,0] - right_eye_center[0,0]
            scale = aligned_eye_distance / eye_distance

            eyes_center = (int((left_eye_center[0,0] + right_eye_center[0,0]) // 2),
                    int((left_eye_center[0,1] + right_eye_center[0,1]) // 2))
                    
            metrix = cv2.getRotationMatrix2D(eyes_center, degree, scale)

            warped = cv2.warpAffine(image_origin, metrix, (image_width, image_height),
                flags=cv2.INTER_CUBIC)

            (startX, endX, startY, endY) = getCropDimension(rect, eyes_center)

            croped = warped[startY:endY, startX:endX]
            output = cv2.resize(croped, OUTPUT_SIZE)
            #output = warped[startY:endY, startX:endX]
            
            output_file = output_path + str(idx+1) + image_type
            cv2.imshow(output_file, output)
            cv2.imwrite(output_file, output)
        
cv2.waitKey(0)   
cv2.destroyAllWindows()
```

![03](https://user-images.githubusercontent.com/105587839/203671224-1fa061ca-663d-4ca1-8dfe-7a933b493de3.png)


EA 마크를 얼굴로 잘못 인식한 경우도 볼 수 있다.



```python
```
