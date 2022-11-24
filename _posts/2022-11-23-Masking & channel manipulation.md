---
layout: single
title:  "03 Masking & channel manipulation"
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



```python
import cv2
import numpy as np
```


```python
img = cv2.imread("A.png") #change image
```


```python
(height, width) = img.shape[:2]
center = (width // 2, height // 2)
```


```python
cv2.imshow("Chunsik", img)
```


```python
mask = np.zeros(img.shape[:2], dtype = "uint8") #bring height n width
```


```python
cv2.imshow("mask", mask)
```


```python
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![01](https://user-images.githubusercontent.com/105587839/203484022-61b93ded-9f6c-492f-9fa2-1577242a9e7c.png)


The mask is still black, so I will correct it.



# Mask



```python
cv2.circle(mask, center, 300, (255,255,255), -1)
```

<pre>
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
</pre>

```python
cv2.imshow("mask", mask)
```


```python
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![02](https://user-images.githubusercontent.com/105587839/203484033-a5aee101-345b-42af-9312-b30b278e993c.png)



## Bitwise_and



```python
masked = cv2.bitwise_and(img,img, mask = mask)
```


```python
cv2.imshow("chunsik with mask", masked)
```


```python
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![03](https://user-images.githubusercontent.com/105587839/203484048-26148eff-f64c-47ac-b548-5086d70dad0f.png)



# Channel manipulation


## Channel

 - digital에서 image를 나타낼 때, 각각 색을 나타냄

 - RGB, HSV etc



```python
(height, width) = img.shape[:2]
center = (width // 2, height // 2)
```


```python
cv2.imshow("Chunsik", img)
```


```python
(Blue, Green, Red) = cv2.split(img)
```


```python
cv2.imshow("Red channel", Red)
cv2.imshow("Green channel", Green)
cv2.imshow("Blue channel", Blue)
```


```python
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![04](https://user-images.githubusercontent.com/105587839/203484065-a0d803ab-7b09-421f-af0b-6598fb90e170.png)



Color가 잘 안나온다.  



```python
zeros = np.zeros(img.shape[:2], dtype = "uint8")
cv2.imshow("Red channel", cv2.merge([zeros, zeros, Red]))
cv2.imshow("Green channel", cv2.merge([zeros, Green, zeros]))
cv2.imshow("Blue channel", cv2.merge([Blue, zeros, zeros]))
```


```python
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![05](https://user-images.githubusercontent.com/105587839/203484074-c71011ba-8fb7-4aca-92d5-718af582ff62.png)



### Other Filters



```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Filter", gray)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Filter", hsv)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB Filter", lab)
```


```python
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![06](https://user-images.githubusercontent.com/105587839/203484086-bcd5398e-bab4-44c7-8d04-6ae35be34e13.png)



