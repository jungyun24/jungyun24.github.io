---
layout: single
title:  "OpenCV Image Convert 1"
categories: Until_YOLO
tag: [coding, opencv, convert, image, python, computervision]
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


# OpenCV image Convert



```python
import cv2
```


```python
import numpy as np
```

## Coordinate system of image pixels


### Check OpenCV version



```python
print("OpenCV version")
print(cv2.__version__)
```

<pre>
OpenCV version
4.6.0
</pre>
### Show image & Grayscale



```python
img = cv2.imread("KJY.jpg")
```


```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```


```python
cv2.imshow("KJY",img)
cv2.imshow("KJY - gray", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img width="1246" alt="KJY_output" src="https://user-images.githubusercontent.com/105587839/203203863-c6fff225-1ca2-41f6-8583-e2f0e542c27b.png">


### Show image shape



```python
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {} pixels".format(img.shape[2]))
```

<pre>
width: 832 pixels
height: 1112 pixels
channels: 3 pixels
</pre>
### Save image



```python
cv2.imwrite("KJY.jpg",img)
```

<pre>
True
</pre>
## Draw on an image


### Check pixel



```python
(b,g,r) = img[0, 0]   #check the value of the (0.0) coordinates
print("Pixel at (0,0) - Red: {}, Green: {}, Blue: {}".format(r,g,b))
```

<pre>
Pixel at (0,0) - Red: 255, Green: 255, Blue: 255
</pre>
### Change Color (make dot)



```python
dot = img[50:100, 50:100]   #Brings 50~100pixels of image's width and height
cv2.imshow("Dot", dot)

img[50:100, 50:100] = (0,0,255)   #Change Red (b,g,r)
cv2.imshow("KJY", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

![KJY_Change](https://user-images.githubusercontent.com/105587839/203203981-66327630-36f0-43ca-a0a2-64cc6ed5a7a7.png)



### Draw rectangle



```python
img = cv2.imread("KJY.jpg")
img = cv2.rectangle(img, (75,75), (150,150), (0,255,0), 3)   

cv2.imshow("rect",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

***cv2.rectangle(*img*,*start*, *end*, *color*, *thickness*)**    

Parameters:

   - img - image

   - start - start coordinate(ex;(0,0))

   - end - end coordinate(ex;(500,500))

   - color - BGR

   - thickness(int)


![KJY_rect](https://user-images.githubusercontent.com/105587839/203204066-cf581cc4-17e0-47b5-af03-a1624de0c47d.png)



### Draw circle



```python
img = cv2.circle(img, (200,75),25, (0,255,255), -1)   

cv2.imshow("circle",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

***cv2.circle(*img*,*center*, *radian*, *color*, *thickness*)**    

Parameters:

   - img - image

   - center - center of circle(x,y)

   - radian - radius

   - color - BGR

   - thickness(int), if you write -1, fill the inside


![KJY_circle](https://user-images.githubusercontent.com/105587839/203204187-39ffaa38-c6d3-4052-ba18-c5d497bc0706.png)



### Draw line



```python
img = cv2.line(img, (350, 100),(400,100), (255, 0, 0), 5)   

cv2.imshow("line",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

***cv2.line(*img*,*start*, *end*, *color*, *thickness*)**    

Parameters:

   - img - image

   - start - start coordinate(ex;(0,0))

   - end - end coordinate(ex;(500,500))

   - color - BGR

   - thickness(int), if you write -1, fill the inside


![KJY_line](https://user-images.githubusercontent.com/105587839/203204283-981defdd-85f6-4beb-81f0-b20ab400e0e5.png)



### Input text



```python
img = cv2.putText(img,'Jungyun Kim', (500,900), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),4)   

cv2.imshow("Text",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

***cv2.putText(*img*,*text*, *org*, *font*, *fontScale*, *color*)**    

Parameters:

   - img - image

   - text - write down what you want

   - org - locate

   - font - font.type. CV2.FONT_XXX

   - fontScale -Font Size

   - color - BGR

![KJY_text](https://user-images.githubusercontent.com/105587839/203204348-ad61a11c-1fc2-405d-b907-36faf0ebc827.png)


