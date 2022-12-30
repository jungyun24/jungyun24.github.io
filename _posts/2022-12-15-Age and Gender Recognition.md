---
layout: single
title:  "06 Age and Gender Recognition"
categories: Until_YOLO
tag: [python, code, age recogniton, recognition, gender recognition]
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


# Age and Gender Recognition



```python
import numpy as np
import cv2
import time
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import tkinter.scrolledtext as tkst

face_model = 'res10_300x300_ssd_iter_140000.caffemodel'
face_prototxt = 'deploy.prototxt.txt'
age_model = 'age_net.caffemodel'
age_prototxt = 'age_deploy.prototxt'
gender_model = 'gender_net.caffemodel'
gender_prototxt = 'gender_deploy.prototxt'
image_file = 'holiday_jeju2.jpg'

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male','Female']

title_name = 'Age and Gender Recognition'
min_confidence = 0.5  #50% 이상이여야 인식으로 본다.
min_likeness = 0.5
frame_count = 0
recognition_count = 0
elapsed_time = 0
OUTPUT_SIZE = (300, 300)

detector = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
age_detector = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
gender_detector = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)

def selectFile():
    file_name =  filedialog.askopenfilename(initialdir = "./image",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print('File name : ', file_name)
    read_image = cv2.imread(file_name)
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    (height, width) = read_image.shape[:2]
    fileLabel['text'] = file_name
    detectAndDisplay(read_image)
    
def detectAndDisplay(image):
    start_time = time.time()
    (h, w) = image.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, OUTPUT_SIZE), 1.0, OUTPUT_SIZE,
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob) #얼굴을 찾아줌
    detections = detector.forward()

    log_ScrolledText.delete(1.0,END)

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX] #새로운 이미지(얼굴)
            (fH, fW) = face.shape[:2]

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),swapRB=False)
            
    
            age_detector.setInput(face_blob)
            age_predictions = age_detector.forward()
            age_index = age_predictions[0].argmax()   #가장 높은 확률을 가져옴
            age = age_list[age_index]
            age_confidence = age_predictions[0][age_index]
            
            gender_detector.setInput(face_blob)
            gender_predictions = gender_detector.forward()
            gender_index = gender_predictions[0].argmax()
            gender = gender_list[gender_index]
            gender_confidence = gender_predictions[0][gender_index]

            text = "{}: {}".format(gender, age)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
            log_ScrolledText.insert(END, "%10s %10s %10.2f %2s" % ('Gender : ', gender, gender_confidence*100, '%')+'\n', 'TITLE')
            log_ScrolledText.insert(END, "%10s %10s %10.2f %2s" % ('Age    : ', age, age_confidence*100, '%')+'\n\n', 'TITLE')
            log_ScrolledText.insert(END, "%15s %20s" % ('Age', 'Probability(%)')+'\n', 'HEADER')
            for i in range(len(age_list)):
                log_ScrolledText.insert(END, "%10s %15.2f" % (age_list[i], age_predictions[0][i]*100)+'\n')
                
            log_ScrolledText.insert(END, "%12s %20s" % ('Gender', 'Probability(%)')+'\n', 'HEADER')
            for i in range(len(gender_list)):
                log_ScrolledText.insert(END, "%10s %15.2f" % (gender_list[i], gender_predictions[0][i]*100)+'\n')
                

                
    frame_time = time.time() - start_time
    global elapsed_time
    elapsed_time += frame_time
    print("Frame {} time {:.3f} seconds".format(frame_count, frame_time))
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    detection.config(image=imgtk)
    detection.image = imgtk
    
    

main = Tk()
main.title(title_name)
main.geometry()

# load the input image and convert it from BGR to RGB
read_image = cv2.imread(image_file)
(height, width) = read_image.shape[:2]
image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
imgtk = ImageTk.PhotoImage(image=image)

label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4) #4개의 격자


fileLabel=Label(main, text=image_file)
fileLabel.grid(row=1,column=0,columnspan=2) #2개의 격자
Button(main,text="File Select", height=2,command=lambda:selectFile()).grid(row=1, column=2, columnspan=2, sticky=(N, S, W, E))
detection=Label(main, image=imgtk)
detection.grid(row=2,column=0,columnspan=4)

log_ScrolledText = tkst.ScrolledText(main, height=20)
log_ScrolledText.grid(row=3,column=0,columnspan=4, sticky=(N, S, W, E))

log_ScrolledText.configure(font='TkFixedFont')

log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14))
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

detectAndDisplay(read_image)

main.mainloop()
```

<pre>
Frame 0 time 0.135 seconds
</pre>

![A1](https://user-images.githubusercontent.com/105587839/207778470-117fdc45-fc6d-4267-b49c-024ff92d3d14.png)



```python
import numpy as np
import cv2
import time

face_model = 'res10_300x300_ssd_iter_140000.caffemodel'
face_prototxt = 'deploy.prototxt.txt'
age_model = 'age_net.caffemodel'
age_prototxt = 'age_deploy.prototxt'
gender_model = 'gender_net.caffemodel'
gender_prototxt = 'gender_deploy.prototxt'

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male','Female']

title_name = 'Age and Gender Recognition'
min_confidence = 0.5
recognition_count = 0
elapsed_time = 0
OUTPUT_SIZE = (300, 300)

detector = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
age_detector = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
gender_detector = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)

    
def detectAndDisplay(image):
    start_time = time.time()
    (h, w) = image.shape[:2]

    
    imageBlob = cv2.dnn.blobFromImage(image, 1.0, OUTPUT_SIZE,
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),swapRB=False)
            
    
            age_detector.setInput(face_blob)
            age_predictions = age_detector.forward()
            age_index = age_predictions[0].argmax()
            age = age_list[age_index]
            age_confidence = age_predictions[0][age_index]
            
            gender_detector.setInput(face_blob)
            gender_predictions = gender_detector.forward()
            gender_index = gender_predictions[0].argmax()
            gender = gender_list[gender_index]
            gender_confidence = gender_predictions[0][gender_index]

            text = "{}: {:.2f}% {}: {:.2f}%".format(gender, gender_confidence*100, age, age_confidence*100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            print('==============================')
            print("Gender {} time {:.2f} %".format(gender, gender_confidence*100))
            print("Age {} time {:.2f} %".format(age, age_confidence*100))
            print("Age     Probability(%)")
            for i in range(len(age_list)):
                print("{}  {:.2f}%".format(age_list[i], age_predictions[0][i]*100))
                
            print("Gender  Probability(%)")
            for i in range(len(gender_list)):
                print("{}  {:.2f} %".format(gender_list[i], gender_predictions[0][i]*100))
                

                
    frame_time = time.time() - start_time
    global elapsed_time
    elapsed_time += frame_time
    print("Frame time {:.3f} seconds".format(frame_time))
    
    cv2.imshow(title_name, image)
    

vs = cv2.VideoCapture(0)
time.sleep(2.0)
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
==============================
Gender Male time 99.04 %
Age (15-20) time 90.79 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.02%
(8-12)  0.76%
(15-20)  90.79%
(25-32)  0.57%
(38-43)  1.78%
(48-53)  6.04%
(60-100)  0.03%
Gender  Probability(%)
Male  99.04 %
Female  0.96 %
Frame time 0.073 seconds
==============================
Gender Male time 98.80 %
Age (15-20) time 90.16 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.02%
(8-12)  0.75%
(15-20)  90.16%
(25-32)  1.01%
(38-43)  2.21%
(48-53)  5.82%
(60-100)  0.03%
Gender  Probability(%)
Male  98.80 %
Female  1.20 %
Frame time 0.077 seconds
==============================
Gender Male time 99.89 %
Age (15-20) time 96.78 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.03%
(8-12)  1.23%
(15-20)  96.78%
(25-32)  0.72%
(38-43)  0.43%
(48-53)  0.80%
(60-100)  0.01%
Gender  Probability(%)
Male  99.89 %
Female  0.11 %
Frame time 0.107 seconds
==============================
Gender Male time 98.94 %
Age (15-20) time 96.04 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.02%
(8-12)  0.49%
(15-20)  96.04%
(25-32)  0.46%
(38-43)  1.10%
(48-53)  1.88%
(60-100)  0.01%
Gender  Probability(%)
Male  98.94 %
Female  1.06 %
Frame time 0.067 seconds
==============================
Gender Male time 97.15 %
Age (15-20) time 97.87 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.01%
(8-12)  0.26%
(15-20)  97.87%
(25-32)  0.37%
(38-43)  0.73%
(48-53)  0.75%
(60-100)  0.01%
Gender  Probability(%)
Male  97.15 %
Female  2.85 %
Frame time 0.075 seconds
==============================
Gender Male time 98.37 %
Age (15-20) time 97.86 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.01%
(8-12)  0.64%
(15-20)  97.86%
(25-32)  0.67%
(38-43)  0.41%
(48-53)  0.40%
(60-100)  0.01%
Gender  Probability(%)
Male  98.37 %
Female  1.63 %
Frame time 0.062 seconds
==============================
Gender Male time 98.88 %
Age (15-20) time 97.96 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.01%
(8-12)  0.44%
(15-20)  97.96%
(25-32)  0.67%
(38-43)  0.44%
(48-53)  0.47%
(60-100)  0.01%
Gender  Probability(%)
Male  98.88 %
Female  1.12 %
Frame time 0.070 seconds
==============================
Gender Male time 98.83 %
Age (15-20) time 94.57 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.03%
(8-12)  1.89%
(15-20)  94.57%
(25-32)  1.88%
(38-43)  0.72%
(48-53)  0.88%
(60-100)  0.02%
Gender  Probability(%)
Male  98.83 %
Female  1.17 %
Frame time 0.061 seconds
==============================
Gender Male time 99.18 %
Age (15-20) time 92.89 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.06%
(8-12)  3.16%
(15-20)  92.89%
(25-32)  2.59%
(38-43)  0.56%
(48-53)  0.71%
(60-100)  0.02%
Gender  Probability(%)
Male  99.18 %
Female  0.82 %
Frame time 0.065 seconds
==============================
Gender Male time 95.69 %
Age (15-20) time 96.96 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.02%
(8-12)  1.68%
(15-20)  96.96%
(25-32)  1.00%
(38-43)  0.19%
(48-53)  0.15%
(60-100)  0.01%
Gender  Probability(%)
Male  95.69 %
Female  4.31 %
Frame time 0.059 seconds
==============================
Gender Male time 95.34 %
Age (15-20) time 97.23 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.02%
(8-12)  1.33%
(15-20)  97.23%
(25-32)  1.05%
(38-43)  0.21%
(48-53)  0.16%
(60-100)  0.01%
Gender  Probability(%)
Male  95.34 %
Female  4.66 %
Frame time 0.064 seconds
==============================
Gender Male time 98.88 %
Age (15-20) time 83.80 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.11%
(8-12)  9.72%
(15-20)  83.80%
(25-32)  2.94%
(38-43)  1.19%
(48-53)  2.20%
(60-100)  0.03%
Gender  Probability(%)
Male  98.88 %
Female  1.12 %
Frame time 0.062 seconds
==============================
Gender Male time 92.22 %
Age (15-20) time 88.12 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.08%
(8-12)  3.47%
(15-20)  88.12%
(25-32)  3.27%
(38-43)  2.24%
(48-53)  2.77%
(60-100)  0.04%
Gender  Probability(%)
Male  92.22 %
Female  7.78 %
Frame time 0.068 seconds
==============================
Gender Male time 99.42 %
Age (15-20) time 79.23 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.09%
(8-12)  9.70%
(15-20)  79.23%
(25-32)  3.52%
(38-43)  2.72%
(48-53)  4.68%
(60-100)  0.05%
Gender  Probability(%)
Male  99.42 %
Female  0.58 %
Frame time 0.059 seconds
==============================
Gender Male time 99.63 %
Age (15-20) time 77.41 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.10%
(8-12)  13.45%
(15-20)  77.41%
(25-32)  3.65%
(38-43)  2.01%
(48-53)  3.32%
(60-100)  0.04%
Gender  Probability(%)
Male  99.63 %
Female  0.37 %
Frame time 0.063 seconds
==============================
Gender Male time 99.57 %
Age (15-20) time 77.39 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.19%
(8-12)  10.90%
(15-20)  77.39%
(25-32)  1.47%
(38-43)  2.61%
(48-53)  7.40%
(60-100)  0.04%
Gender  Probability(%)
Male  99.57 %
Female  0.43 %
Frame time 0.058 seconds
==============================
Gender Male time 99.82 %
Age (15-20) time 71.70 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.39%
(8-12)  17.94%
(15-20)  71.70%
(25-32)  1.29%
(38-43)  2.03%
(48-53)  6.61%
(60-100)  0.04%
Gender  Probability(%)
Male  99.82 %
Female  0.18 %
Frame time 0.071 seconds
==============================
Gender Male time 99.82 %
Age (15-20) time 64.02 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.46%
(8-12)  22.43%
(15-20)  64.02%
(25-32)  1.51%
(38-43)  3.32%
(48-53)  8.22%
(60-100)  0.04%
Gender  Probability(%)
Male  99.82 %
Female  0.18 %
Frame time 0.063 seconds
==============================
Gender Male time 99.91 %
Age (15-20) time 56.61 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.62%
(8-12)  28.62%
(15-20)  56.61%
(25-32)  4.49%
(38-43)  2.84%
(48-53)  6.75%
(60-100)  0.05%
Gender  Probability(%)
Male  99.91 %
Female  0.09 %
Frame time 0.058 seconds
==============================
Gender Male time 99.90 %
Age (15-20) time 58.82 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.30%
(8-12)  25.30%
(15-20)  58.82%
(25-32)  5.48%
(38-43)  3.11%
(48-53)  6.91%
(60-100)  0.06%
Gender  Probability(%)
Male  99.90 %
Female  0.10 %
Frame time 0.061 seconds
==============================
Gender Male time 99.72 %
Age (15-20) time 68.86 %
Age     Probability(%)
(0-2)  0.03%
(4-6)  0.31%
(8-12)  12.42%
(15-20)  68.86%
(25-32)  5.38%
(38-43)  3.63%
(48-53)  9.29%
(60-100)  0.08%
Gender  Probability(%)
Male  99.72 %
Female  0.28 %
Frame time 0.064 seconds
==============================
Gender Male time 99.81 %
Age (15-20) time 94.67 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.02%
(8-12)  0.70%
(15-20)  94.67%
(25-32)  0.27%
(38-43)  1.38%
(48-53)  2.94%
(60-100)  0.01%
Gender  Probability(%)
Male  99.81 %
Female  0.19 %
Frame time 0.063 seconds
==============================
Gender Male time 99.86 %
Age (15-20) time 97.05 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.02%
(8-12)  0.62%
(15-20)  97.05%
(25-32)  0.24%
(38-43)  0.83%
(48-53)  1.23%
(60-100)  0.01%
Gender  Probability(%)
Male  99.86 %
Female  0.14 %
Frame time 0.059 seconds
==============================
Gender Male time 99.93 %
Age (15-20) time 95.57 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.05%
(8-12)  1.35%
(15-20)  95.57%
(25-32)  0.59%
(38-43)  0.93%
(48-53)  1.50%
(60-100)  0.01%
Gender  Probability(%)
Male  99.93 %
Female  0.07 %
Frame time 0.063 seconds
==============================
Gender Male time 99.89 %
Age (15-20) time 47.16 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.04%
(8-12)  1.35%
(15-20)  47.16%
(25-32)  1.74%
(38-43)  10.54%
(48-53)  39.06%
(60-100)  0.10%
Gender  Probability(%)
Male  99.89 %
Female  0.11 %
Frame time 0.058 seconds
==============================
Gender Male time 99.91 %
Age (15-20) time 94.38 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.07%
(8-12)  1.00%
(15-20)  94.38%
(25-32)  0.65%
(38-43)  1.46%
(48-53)  2.41%
(60-100)  0.02%
Gender  Probability(%)
Male  99.91 %
Female  0.09 %
Frame time 0.060 seconds
==============================
Gender Male time 99.47 %
Age (15-20) time 84.58 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.13%
(8-12)  2.55%
(15-20)  84.58%
(25-32)  1.96%
(38-43)  3.04%
(48-53)  7.66%
(60-100)  0.05%
Gender  Probability(%)
Male  99.47 %
Female  0.53 %
Frame time 0.059 seconds
==============================
Gender Male time 99.62 %
Age (15-20) time 83.68 %
Age     Probability(%)
(0-2)  0.03%
(4-6)  0.21%
(8-12)  3.37%
(15-20)  83.68%
(25-32)  2.57%
(38-43)  3.38%
(48-53)  6.68%
(60-100)  0.06%
Gender  Probability(%)
Male  99.62 %
Female  0.38 %
Frame time 0.062 seconds
==============================
Gender Male time 99.52 %
Age (15-20) time 86.35 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.17%
(8-12)  2.87%
(15-20)  86.35%
(25-32)  1.55%
(38-43)  2.30%
(48-53)  6.70%
(60-100)  0.04%
Gender  Probability(%)
Male  99.52 %
Female  0.48 %
Frame time 0.060 seconds
</pre>
<pre>
==============================
Gender Male time 99.67 %
Age (15-20) time 88.45 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.12%
(8-12)  2.82%
(15-20)  88.45%
(25-32)  1.21%
(38-43)  1.75%
(48-53)  5.60%
(60-100)  0.03%
Gender  Probability(%)
Male  99.67 %
Female  0.33 %
Frame time 0.096 seconds
==============================
Gender Male time 99.72 %
Age (15-20) time 80.13 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.06%
(8-12)  1.83%
(15-20)  80.13%
(25-32)  1.69%
(38-43)  4.66%
(48-53)  11.56%
(60-100)  0.06%
Gender  Probability(%)
Male  99.72 %
Female  0.28 %
Frame time 0.077 seconds
==============================
Gender Male time 99.82 %
Age (15-20) time 94.52 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.04%
(8-12)  1.51%
(15-20)  94.52%
(25-32)  0.60%
(38-43)  1.08%
(48-53)  2.23%
(60-100)  0.02%
Gender  Probability(%)
Male  99.82 %
Female  0.18 %
Frame time 0.070 seconds
==============================
Gender Male time 99.72 %
Age (15-20) time 81.13 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.04%
(8-12)  1.05%
(15-20)  81.13%
(25-32)  1.75%
(38-43)  6.48%
(48-53)  9.46%
(60-100)  0.07%
Gender  Probability(%)
Male  99.72 %
Female  0.28 %
Frame time 0.081 seconds
==============================
Gender Male time 99.85 %
Age (15-20) time 96.65 %
Age     Probability(%)
(0-2)  0.00%
(4-6)  0.01%
(8-12)  0.50%
(15-20)  96.65%
(25-32)  0.44%
(38-43)  1.27%
(48-53)  1.11%
(60-100)  0.01%
Gender  Probability(%)
Male  99.85 %
Female  0.15 %
Frame time 0.059 seconds
==============================
Gender Male time 99.96 %
Age (15-20) time 95.20 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.03%
(8-12)  0.93%
(15-20)  95.20%
(25-32)  0.99%
(38-43)  1.30%
(48-53)  1.52%
(60-100)  0.02%
Gender  Probability(%)
Male  99.96 %
Female  0.04 %
Frame time 0.063 seconds
==============================
Gender Male time 99.98 %
Age (15-20) time 59.51 %
Age     Probability(%)
(0-2)  0.03%
(4-6)  0.46%
(8-12)  14.64%
(15-20)  59.51%
(25-32)  8.96%
(38-43)  5.13%
(48-53)  11.17%
(60-100)  0.10%
Gender  Probability(%)
Male  99.98 %
Female  0.02 %
Frame time 0.059 seconds
==============================
Gender Male time 99.97 %
Age (15-20) time 70.51 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.40%
(8-12)  12.42%
(15-20)  70.51%
(25-32)  4.74%
(38-43)  3.91%
(48-53)  7.93%
(60-100)  0.06%
Gender  Probability(%)
Male  99.97 %
Female  0.03 %
Frame time 0.062 seconds
==============================
Gender Male time 99.91 %
Age (15-20) time 83.71 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.18%
(8-12)  9.29%
(15-20)  83.71%
(25-32)  4.46%
(38-43)  1.02%
(48-53)  1.30%
(60-100)  0.03%
Gender  Probability(%)
Male  99.91 %
Female  0.09 %
Frame time 0.058 seconds
==============================
Gender Male time 99.98 %
Age (15-20) time 83.89 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.07%
(8-12)  3.73%
(15-20)  83.89%
(25-32)  7.23%
(38-43)  2.46%
(48-53)  2.55%
(60-100)  0.04%
Gender  Probability(%)
Male  99.98 %
Female  0.02 %
Frame time 0.063 seconds
==============================
Gender Male time 99.97 %
Age (15-20) time 76.71 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.09%
(8-12)  4.62%
(15-20)  76.71%
(25-32)  8.95%
(38-43)  4.32%
(48-53)  5.22%
(60-100)  0.06%
Gender  Probability(%)
Male  99.97 %
Female  0.03 %
Frame time 0.066 seconds
==============================
Gender Male time 99.99 %
Age (15-20) time 71.03 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.11%
(8-12)  6.45%
(15-20)  71.03%
(25-32)  9.78%
(38-43)  3.89%
(48-53)  8.66%
(60-100)  0.07%
Gender  Probability(%)
Male  99.99 %
Female  0.01 %
Frame time 0.060 seconds
==============================
Gender Male time 99.99 %
Age (15-20) time 71.68 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.09%
(8-12)  5.46%
(15-20)  71.68%
(25-32)  10.17%
(38-43)  4.34%
(48-53)  8.16%
(60-100)  0.07%
Gender  Probability(%)
Male  99.99 %
Female  0.01 %
Frame time 0.059 seconds
==============================
Gender Male time 99.98 %
Age (15-20) time 88.87 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.09%
(8-12)  1.28%
(15-20)  88.87%
(25-32)  1.43%
(38-43)  3.07%
(48-53)  5.22%
(60-100)  0.04%
Gender  Probability(%)
Male  99.98 %
Female  0.02 %
Frame time 0.061 seconds
==============================
Gender Male time 99.98 %
Age (15-20) time 88.08 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.10%
(8-12)  1.47%
(15-20)  88.08%
(25-32)  1.21%
(38-43)  3.21%
(48-53)  5.89%
(60-100)  0.04%
Gender  Probability(%)
Male  99.98 %
Female  0.02 %
Frame time 0.058 seconds
==============================
Gender Male time 99.98 %
Age (15-20) time 86.16 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.13%
(8-12)  3.33%
(15-20)  86.16%
(25-32)  2.70%
(38-43)  3.30%
(48-53)  4.34%
(60-100)  0.04%
Gender  Probability(%)
Male  99.98 %
Female  0.02 %
Frame time 0.063 seconds
==============================
Gender Male time 99.97 %
Age (15-20) time 82.24 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.10%
(8-12)  3.38%
(15-20)  82.24%
(25-32)  4.33%
(38-43)  4.50%
(48-53)  5.39%
(60-100)  0.05%
Gender  Probability(%)
Male  99.97 %
Female  0.03 %
Frame time 0.059 seconds
==============================
Gender Male time 99.97 %
Age (15-20) time 84.03 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.09%
(8-12)  3.25%
(15-20)  84.03%
(25-32)  3.13%
(38-43)  4.18%
(48-53)  5.27%
(60-100)  0.04%
Gender  Probability(%)
Male  99.97 %
Female  0.03 %
Frame time 0.062 seconds
==============================
Gender Male time 99.96 %
Age (15-20) time 75.13 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.10%
(8-12)  3.89%
(15-20)  75.13%
(25-32)  5.80%
(38-43)  6.69%
(48-53)  8.30%
(60-100)  0.07%
Gender  Probability(%)
Male  99.96 %
Female  0.04 %
Frame time 0.062 seconds
==============================
Gender Male time 99.95 %
Age (15-20) time 77.24 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.11%
(8-12)  3.89%
(15-20)  77.24%
(25-32)  2.99%
(38-43)  5.73%
(48-53)  9.95%
(60-100)  0.07%
Gender  Probability(%)
Male  99.95 %
Female  0.05 %
Frame time 0.055 seconds
==============================
Gender Male time 99.95 %
Age (15-20) time 78.26 %
Age     Probability(%)
(0-2)  0.01%
(4-6)  0.12%
(8-12)  3.62%
(15-20)  78.26%
(25-32)  2.60%
(38-43)  5.94%
(48-53)  9.38%
(60-100)  0.06%
Gender  Probability(%)
Male  99.95 %
Female  0.05 %
Frame time 0.061 seconds
==============================
Gender Male time 99.97 %
Age (15-20) time 70.44 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.14%
(8-12)  6.12%
(15-20)  70.44%
(25-32)  5.00%
(38-43)  7.02%
(48-53)  11.20%
(60-100)  0.07%
Gender  Probability(%)
Male  99.97 %
Female  0.03 %
Frame time 0.063 seconds
==============================
Gender Male time 99.95 %
Age (15-20) time 78.00 %
Age     Probability(%)
(0-2)  0.02%
(4-6)  0.10%
(8-12)  3.09%
(15-20)  78.00%
(25-32)  2.62%
(38-43)  5.65%
(48-53)  10.44%
(60-100)  0.07%
Gender  Probability(%)
Male  99.95 %
Female  0.05 %
Frame time 0.060 seconds
</pre>
![A2](https://user-images.githubusercontent.com/105587839/207778534-e47027ab-50e5-46ee-b314-d65659153939.png)

