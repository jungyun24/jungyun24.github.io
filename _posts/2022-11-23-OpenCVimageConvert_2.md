---
layout: single
title:  "OpenCV Image Convert 2"
categories: Until_YOLO
tag: [coding, opencv, convert, image, python, computervision]
toc: true
author_profile: false
sidebar:
  nav: "docs"
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
```


```python
import numpy as np
```

 # Image Transform



```python
img = cv2.imread("KJY.jpg")
```


```python
(height, width) = img.shape[:2] #[0] in height, [1] in width
center = (width // 2, height // 2)
```

## Image Move



```python
move = np.float32([[1,0,100], [0,1,100]])
moved = cv2.warpAffine(img, move, (width, height))
cv2.imshow("Moved down: +, up: - and right: +, left: -",moved)
```


```python
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![](jpg)


## Image Rotation



```python
move = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated = cv2.warpAffine(img, move, (width, height))
cv2.imshow("Rotated degrees", rotated)
```


```python
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![](02)


## Resize



```python
ratio = 200.0 / width
dimension = (200, int(height * ratio))
```


```python
resized = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA) #enlarge ; inter_linear; collapse : inter_area... etc
cv2.imshow("Resized", resized)
```


```python
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![](03)


## Symmetry



```python
flipped = cv2.flip(img, -1)
cv2.imshow("Flipped Horizontal 1, Vertical 0, both -1", flipped)
```


```python
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![](04)



```python
```
