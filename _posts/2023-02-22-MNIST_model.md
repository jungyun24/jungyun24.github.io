---
layout: single
title:  "03 MNIST"
categories: PyTorch
tag: [Pytorch,mnist]
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


# 파이토치 MNIST 모델
<iframe width="560" height="315" src="https://www.youtube.com/embed/IwLOWwrz26w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
Jupyter Notebook 자료 : https://drive.google.com/file/d/1EFnm57qVjID9cU-FbX1Hy-vgNDqq5iqe/view?usp=sharing
## modules import 



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
```

## 전처리 설정

- `transform.Compose`



```python
transform = transforms.Compose([transforms.ToTensor(),  #PIL iamge를 tensor로
                               transforms.Normalize((0.5,),(0.5,))])
```

## 데이터 로드 및 데이터 확인



```python
trainset = dsets.MNIST(root='MNIST_data/',
                      train = True,
                      download = True,
                      transform = transform)
testset = dsets.MNIST(root='MNIST_data/',
                      train = False,
                      download = True,
                      transform = transform)
```


```python
train_loader = DataLoader(trainset, batch_size = 128, shuffle = True, num_workers=2)
test_loader = DataLoader(testset, batch_size = 128, shuffle = False, num_workers=2)
```


```python
image, label = next(iter(train_loader))
```


```python
image.shape, label.shape
```

<pre>
(torch.Size([128, 1, 28, 28]), torch.Size([128]))
</pre>

```python
def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    fig = plt.figure(figsize =(10,5))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
```


```python
dataiter = iter(train_loader)
images,labels = next(dataiter)
imshow(torchvision.utils.make_grid(images[:4]))   #make_grid는 알아서 보여줌
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAzoAAAD5CAYAAADvJUAzAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAm/klEQVR4nO3de3BUZZ7/8U9zawKEdgOSiwnZuAaYSZBRUIabIELKqKCCM15GgXXLGxDNMIhAUMAqCGDJwA7ILK6FUIIwjjATL4OERYIMyxq5CCs6wTFAVGJGJBcCJkDO748t+2fLc1pOcjoJJ+9X1amyv895nvONTyfkm9P9bZ9lWZYAAAAAwENaNXUCAAAAAOA2Ch0AAAAAnkOhAwAAAMBzKHQAAAAAeA6FDgAAAADPodABAAAA4DkUOgAAAAA8h0IHAAAAgOdQ6AAAAADwHAodAAAAAJ7TJlILv/DCC3ruued0/PhxpaWlacmSJRoyZMiPzqurq9OXX36p6Oho+Xy+SKUHAAAAoJmzLEtVVVVKSEhQq1bO7tFEpNDZsGGDsrOz9cILL2jQoEH6j//4D2VmZurQoUPq3r172LlffvmlkpKSIpEWAAAAgEtQSUmJEhMTHc3xWZZluZ1I//79de2112rFihXB2E9+8hPdcccdys3NDTu3oqJCl112mX7961/L7/e7nRoAAACAS0RNTY1++9vfqry8XIFAwNFc1+/o1NbWas+ePZo+fXpIPCMjQ7t27brg/JqaGtXU1AQfV1VVSZL8fj+FDgAAAIB6vaXF9WYEX3/9tc6fP6/Y2NiQeGxsrEpLSy84Pzc3V4FAIHjwsjUAAAAADRWxrms/rLosyzJWYjNmzFBFRUXwKCkpiVRKAAAAAFoI11+61rVrV7Vu3fqCuzdlZWUX3OWReIkaAAAAAPe5fkenXbt26tu3r/Lz80Pi+fn5GjhwoNuXAwAAAIALRKS99JQpU/TAAw+oX79+GjBggFauXKljx47p0UcfjcTlAAAAACBERAqdu+++WydOnNCzzz6r48ePKz09XW+//baSk5MjcTkAAAAACBGRQkeSJk6cqIkTJ0ZqeQAAAACwFbGuawAAAADQVCh0AAAAAHhOxF661tjmzp3b1CkADTZ79mxH5/O8hxc4fd5LPPfhDfzMR0tUn5/59cUdHQAAAACeQ6EDAAAAwHModAAAAAB4DoUOAAAAAM+h0AEAAADgORQ6AAAAADyHQgcAAACA51DoAAAAAPAcCh0AAAAAnkOhAwAAAMBzKHQAAAAAeA6FDgAAAADPodABAAAA4DkUOgAAAAA8h0IHAAAAgOdQ6AAAAADwHAodAAAAAJ5DoQMAAADAc1wvdObMmSOfzxdyxMXFuX0ZAAAAALDVJhKLpqWlaevWrcHHrVu3jsRlAAAAAMAoIoVOmzZtuIsDAAAAoMlE5D06hw8fVkJCglJSUnTPPffos88+sz23pqZGlZWVIQcAAAAANITrhU7//v21Zs0avfPOO3rxxRdVWlqqgQMH6sSJE8bzc3NzFQgEgkdSUpLbKQEAAABoYVwvdDIzMzV27Fj17t1bI0aM0FtvvSVJWr16tfH8GTNmqKKiIniUlJS4nRIAAACAFiYi79H5vo4dO6p37946fPiwcdzv98vv90c6DQAA0ET++te/GuNdunSxndOrV69IpYMWpFUr89/03377bWO8trbWdq3Ro0cb43a/x86ePdt2rYEDBxrjb7zxhjG+dOlS27XOnTtnO9bSRfxzdGpqavTxxx8rPj4+0pcCAAAAAEkRKHSmTp2qgoICFRcX63/+53901113qbKyUuPHj3f7UgAAAABg5PpL1z7//HPde++9+vrrr3X55Zfr5z//uXbv3q3k5GS3LwUAAAAARq4XOuvXr3d7SQAAAABwJOLv0QEAAACAxkahAwAAAMBzIt5eGgAAeN9NN91kO3bdddcZ461bt7adM3bsWGP89ddfd5YYPC/c8+jpp582xjMyMozxbdu22a7Vtm1bYzwvL88YHzlypO1adm644QZj/IsvvrCdw9tG7HFHBwAAAIDnUOgAAAAA8BwKHQAAAACeQ6EDAAAAwHModAAAAAB4Dl3XAACIsE6dOtmO2XUr27dvnzF+7NgxV3JyW7ivMVxXLDuBQKAh6aAF6d27t+3YM88842ita6+91nYsISHBGO/Tp4+ja9THb37zG9uxN954wxivrq6OVDqXDO7oAAAAAPAcCh0AAAAAnkOhAwAAAMBzKHQAAAAAeA6FDgAAAADPoetaC3XNNdcY40eOHLGdc/LkyQhl8+OSk5Ntx4YPH26MDx482BgfO3as7Vrl5eXGeFpamu0cupq4o7i42Bjv3r277Zz6dHJqru666y5jPFwnK6fsunh9+OGHrl0DZk8++aTt2KxZs4xxu58tw4YNs11r7969jvJqzui6hh9KTU01xvPy8ly7xpo1a2zHjh49aoyPGTPGGM/JybFdKz093RhPSkoyxvv27Wu71q233mqM/+EPf7Cd01JwRwcAAACA51DoAAAAAPAcCh0AAAAAnkOhAwAAAMBzKHQAAAAAeA6FDgAAAADPcdxeeseOHXruuee0Z88eHT9+XJs2bdIdd9wRHLcsS3PnztXKlSt18uRJ9e/fX8uXLw/bnhcN4/f7bccmTJhgjC9dutQY/8c//mG71qlTp4zxoqIiY7xnz562a1mWZTtmcvnll9uOxcTEGOM+n88YP3/+vO1av/nNb4zxM2fOhMkOPxSu7XNiYqIx3qqV+e8udXV1tmvZtR3v2rWr7ZznnnvOdswpu+eY0+e3JF1//fXGuN33t93/L8n+/5lda9Rjx47ZrvXSSy8Z42vXrrWdgwtt3brVdsyuvbRda/Ff/OIXtmt5qb20Xcve3/72t42cCZqLe++91xi3+3clnHXr1hnjM2fOdLzWrl27jHG7ts+SfbvowsJCx9e3+5lAe+l63NGprq5Wnz59tGzZMuP4okWLtHjxYi1btkyFhYWKi4vTyJEjVVVV1eBkAQAAAOBiOL6jk5mZqczMTOOYZVlasmSJcnJygn+JWb16tWJjY7Vu3To98sgjDcsWAAAAAC6Cq+/RKS4uVmlpqTIyMoIxv9+voUOH2t7Wq6mpUWVlZcgBAAAAAA3haqFTWloqSYqNjQ2Jx8bGBsd+KDc3V4FAIHgkJSW5mRIAAACAFigiXdd++AZdy7Js37Q7Y8YMVVRUBI+SkpJIpAQAAACgBXH8Hp1w4uLiJP3fnZ34+PhgvKys7IK7PN/x+/1hu4bhx82YMcN27JlnnjHGDx8+bIxHR0fbrmXXRS1cd7WmZNf5KlzXLbsOU3AmKyvLdszNrmefffaZMR6uU5ub6tMprinZdamzi0vSN998Y4zTdc2ZsrIy2zG7PwQ6jTdn9cn5Uvw64Y7evXsb4+H+bXFqw4YNxvjp06ddu0Y4dv9+2f1+lpqaGsl0PMvVOzopKSmKi4tTfn5+MFZbW6uCggINHDjQzUsBAAAAgC3Hd3ROnTqlTz/9NPi4uLhY+/fvV0xMjLp3767s7GzNnz9fqampSk1N1fz589WhQwfdd999riYOAAAAAHYcFzoffPCBbrzxxuDjKVOmSJLGjx+vl19+WdOmTdOZM2c0ceLE4AeGbtmyJexLogAAAADATY4LnWHDhoX91G+fz6c5c+Zozpw5DckLAAAAAOotIl3XAAAAAKApudp1DZF1+eWXG+P/9m//ZjvnzJkzxviTTz5pjG/bts12rXbt2oXJzh1paWnG+DvvvGM7p3379sb4/PnzjXG7TnRwz+jRox3PKS8vN8Y//PBD2zmrVq0yxpcuXWo7JxAIOMqrqX3yySfGeLguXuHuupvY/TyQpKKiIkdrwezqq6+2HXO6X07Pbw7qk/Ol+HXCHdOnTzfGu3Tp4nit//zP/zTG3377bcdruenkyZPG+K9+9Stj/P33349kOp7FHR0AAAAAnkOhAwAAAMBzKHQAAAAAeA6FDgAAAADPodABAAAA4DkUOgAAAAA8h/bSl5Bx48YZ41dccYXtnJdfftkYz8vLcyOlevP7/cZ4Tk6OMW7XQlqSFi9ebIzPmjXLeWJwZNCgQcb4e++9ZztnyJAhxrhdG+kRI0Y4TywMu+8JNy1cuNAYr0+r5p07dxrjn332meO10HT69u3b1CkAzcq1115rOzZq1ChHa4X7efjwww87WqupnT17tqlT8BTu6AAAAADwHAodAAAAAJ5DoQMAAADAcyh0AAAAAHgOhQ4AAAAAz6Hr2iVk7Nixjuds3LgxApk03GuvvWaMZ2RkGOM7duywXWvq1Kmu5ATnJkyYYIw/+OCDjteKj493dI36euihh1xb6/Dhw8Z4dHS0MX78+HHbtey6zgHApaxjx47G+MyZM23ndOrUyRj/6quvjHGnXdqas9tuu62pU/AU7ugAAAAA8BwKHQAAAACeQ6EDAAAAwHModAAAAAB4DoUOAAAAAM9xXOjs2LFDo0aNUkJCgnw+n/70pz+FjE+YMEE+ny/k+PnPf+5WvgAAAADwoxy3l66urlafPn30r//6r7btjm+++WatWrUq+Lhdu3b1z7AFSktLM8avvvpqY/zAgQO2a7355puu5FQf4doC27WRtmvXe88997iRElxm10a6rq7O8Vo9evQwxl988UXbOa1amf9WU5/r18c//vEPY9zuZ155ebntWseOHTPGFy5caIy/88474ZNDs0LLWLRUQ4YMMcbHjBnjeK2VK1ca4x9//LHjtZpa165djfHHHnuskTPxNseFTmZmpjIzM8Oe4/f7FRcXV++kAAAAAKAhIvIene3bt6tbt27q0aOHHnroIZWVldmeW1NTo8rKypADAAAAABrC9UInMzNTa9eu1bZt2/T888+rsLBQw4cPV01NjfH83NxcBQKB4JGUlOR2SgAAAABaGMcvXfsxd999d/C/09PT1a9fPyUnJ+utt94yvh5zxowZmjJlSvBxZWUlxQ4AAACABnG90Pmh+Ph4JScn277J3O/3y+/3RzoNAAAAAC1IxAudEydOqKSkRPHx8ZG+lGfMmzfPGO/QoYMxvnHjxkim86O6d+9ujM+dO9d2ztGjR43xnJwcY7y0tNR5Yoi4H7aX/87o0aMbN5EmEhsba4zbdX0LBAK2ayUnJxvjL7/8suO80PzYPVckyefzOYrbdftrzuy+FrfnoPm56qqrXFvr73//u2trNTW7TroJCQmO11q7dm1D0/Esx4XOqVOn9OmnnwYfFxcXa//+/YqJiVFMTIzmzJmjsWPHKj4+XkeOHNHMmTPVtWtX3Xnnna4mDgAAAAB2HBc6H3zwgW688cbg4+/eXzN+/HitWLFCBw8e1Jo1a1ReXq74+HjdeOON2rBhg6Kjo93LGgAAAADCcFzoDBs2TJZl2Y7zIXYAAAAAmlpEPkcHAAAAAJoShQ4AAAAAz4l41zWYhWupbdfFzO5DV/Py8lzJ6ce0a9fOGH/11VeN8XCfhzRr1ixj/I9//KPzxNBkxo0bZ4z37NnTds6iRYtcu75dV6ZwL6+188knnxjj4fJdtWqVo+v/8z//s+1adl3X7K7fu3dv27WefPJJ2zE0jXDPSafP1yFDhtiOvfvuu8Z4cXGxMV5eXu7o2uEUFRXZjp0+fdoYj4qKsp3TunVrY7xVK/PfaO26HaJpmT5D8cfY/Txev359Q9NpVNdff73tmNPfd8J1n92/f7+jtVoS7ugAAAAA8BwKHQAAAACeQ6EDAAAAwHModAAAAAB4DoUOAAAAAM+h0AEAAADgObSXbiJ2raIlaefOncb4kiVLjPHGaiuYlZVljA8YMMAYf/31123XWrx4sSs5oWlVV1cb43v37rWdM2LEiEil0+huuukmR+fPnj3bdsyu5XqXLl0cxeF9o0ePdjx26tQpY3zbtm22a+3atcsYP3r0qDHer18/27XsWkWH079/f2Pc7vsuPz/f8TXgnp/97GfG+KBBgxyvdeeddxrjtbW1jtdqSnPnzrUdu+yyyxyttWbNGtuxI0eOOFqrJeGODgAAAADPodABAAAA4DkUOgAAAAA8h0IHAAAAgOdQ6AAAAADwHLquNUOPP/54k107JibGduzpp592tNbq1attx7799ltHawFe4PP5bMdatXL2d6dwa6H5mTx5su2YXTe++Ph4YzwQCDi+fqdOnYzxcB3cbr/9dmPcsizH13fT8OHDjXG6rjUtu254bdu2dbyWXUfPxhCuQ+B1111njP/lL38xxuvzvVpXV2eMb9261fFa4I4OAAAAAA+i0AEAAADgORQ6AAAAADyHQgcAAACA51DoAAAAAPAcR13XcnNztXHjRn3yySeKiorSwIEDtXDhQvXs2TN4jmVZmjt3rlauXKmTJ0+qf//+Wr58udLS0lxPHvUXFRVljL/xxhu2czp37myMz5kzxxh/8803HecFeIFd1x677yHJvtOOnabufAVn/vCHPzge+5d/+Rdj/LHHHrNda+zYscZ49+7dw2R3afn000+bOgUYDBgwwNH5p0+fth07f/58Q9MJatPG/KuuXYdbu+5xknTLLbcY4/X5eWz3Nc6bN88Yp+ta/Ti6o1NQUKBJkyZp9+7dys/P17lz55SRkRHSBnDRokVavHixli1bpsLCQsXFxWnkyJGqqqpyPXkAAAAAMHF0R2fz5s0hj1etWqVu3bppz549uuGGG2RZlpYsWaKcnByNGTNG0v99lkpsbKzWrVunRx55xL3MAQAAAMBGg96jU1FRIen/f8hkcXGxSktLlZGRETzH7/dr6NCh2rVrl3GNmpoaVVZWhhwAAAAA0BD1LnQsy9KUKVM0ePBgpaenS5JKS0slSbGxsSHnxsbGBsd+KDc3V4FAIHgkJSXVNyUAAAAAkNSAQmfy5Mk6cOCAXn311QvGfD5fyGPLsi6IfWfGjBmqqKgIHiUlJfVNCQAAAAAkOXyPzneysrKUl5enHTt2KDExMRiPi4uT9H93duLj44PxsrKyC+7yfMfv98vv99cnDQAAAAAwclToWJalrKwsbdq0Sdu3b1dKSkrIeEpKiuLi4pSfn69rrrlGklRbW6uCggItXLjQvazRYL169TLGw7WHtHuf1e9+9ztXcgK84vt/APq+rKws167x5z//2bW10Dz9/e9/N8anTp1qOyfcmFtSU1ON8TvuuMN2Tn1+B7D7N+ell15yvBaan6+//tp27KqrrjLGu3XrZoyPGDHCdq3hw4cb45mZmWGyM3Ozrf++ffuMcbuP7ED9OCp0Jk2apHXr1unPf/6zoqOjg++7CQQCioqKks/nU3Z2tubPn6/U1FSlpqZq/vz56tChg+67776IfAEAAAAA8EOOCp0VK1ZIkoYNGxYSX7VqlSZMmCBJmjZtms6cOaOJEycGPzB0y5Ytio6OdiVhAAAAAPgxjl+69mN8Pp/mzJnDrTcAAAAATaZBn6MDAAAAAM0RhQ4AAAAAz6lXe2lcOpKTk43xN9980/Fas2bNMsZPnjzpeC3AC+za5r/44ouuXWPatGnGOF3X0FQOHz5sjBcVFdnOqU+3Kjc7XCHyjh075uj87t27244VFBQ0NJ1GZfcZkIcOHbKd8/DDD0cqHXwPd3QAAAAAeA6FDgAAAADPodABAAAA4DkUOgAAAAA8h0IHAAAAgOdQ6AAAAADwHNpLe9wVV1xhjMfHxxvj7733nu1aO3fudCUnwCtee+01Y3zAgAGuXePEiROurQUAkfLqq68a49nZ2Y2bSAQdPXrUGL/55puN8b/97W+RTAcXgTs6AAAAADyHQgcAAACA51DoAAAAAPAcCh0AAAAAnkOhAwAAAMBz6LrmAd26dbMde/nll43xU6dOGeP333+/7Vrnzp1zlBfgRHFxsTG+ZcsW2zmPPPKIa9e/6667jPHHH3/cds6QIUOM8bq6OsfX/+Mf/2iMf/rpp47XAoDGduDAAWN85cqVxvjDDz8cyXTqLS8vz3Zs+vTpxjjd1Zov7ugAAAAA8BwKHQAAAACeQ6EDAAAAwHModAAAAAB4DoUOAAAAAM9x1HUtNzdXGzdu1CeffKKoqCgNHDhQCxcuVM+ePYPnTJgwQatXrw6Z179/f+3evdudjHEBu25RknTVVVcZ49u2bTPGS0pKXMkJcKqsrMwYf/DBB23n9OjRwxi3LMvx9a+//npj3O/3286x665WW1trjC9fvtx2rZycHGO8pqbGdg7QnBw5csR27NtvvzXG27dvbzsnNjbWGO/cubMxXllZaZ8cIs7uZ9XixYuN8fj4eNu1Ro0a5UpOkvTWW28Z43PnzjXG9+3bZ7vW+fPnXckJjcfRHZ2CggJNmjRJu3fvVn5+vs6dO6eMjAxVV1eHnHfzzTfr+PHjwePtt992NWkAAAAACMfRHZ3NmzeHPF61apW6deumPXv26IYbbgjG/X6/4uLi3MkQAAAAABxq0Ht0KioqJEkxMTEh8e3bt6tbt27q0aOHHnroIduXpEj/d6uzsrIy5AAAAACAhqh3oWNZlqZMmaLBgwcrPT09GM/MzNTatWu1bds2Pf/88yosLNTw4cNtX7uZm5urQCAQPJKSkuqbEgAAAABIcvjSte+bPHmyDhw4oJ07d4bE77777uB/p6enq1+/fkpOTtZbb72lMWPGXLDOjBkzNGXKlODjyspKih0AAAAADVKvQicrK0t5eXnasWOHEhMTw54bHx+v5ORkHT582Dju9/vDdjUCAAAAAKccFTqWZSkrK0ubNm3S9u3blZKS8qNzTpw4oZKSkrBtBHFxBgwYYIw/++yztnO++eYbY3zWrFmu5AS45Ze//KUx/vrrr9vO+X4TlO+za/vcWOzaSE+dOrWRMwEaz4cffmg71rFjx0bMBM1JUVGRMX777bc3ciZoiRy9R2fSpEl65ZVXtG7dOkVHR6u0tFSlpaU6c+aMJOnUqVOaOnWq/vu//1tHjhzR9u3bNWrUKHXt2lV33nlnRL4AAAAAAPghR3d0VqxYIUkaNmxYSHzVqlWaMGGCWrdurYMHD2rNmjUqLy9XfHy8brzxRm3YsEHR0dGuJQ0AAAAA4Th+6Vo4UVFReueddxqUEAAAAAA0VIM+RwcAAAAAmiMKHQAAAACeU+/P0UHjGz9+vDEeExNjO2fp0qXG+O7du13JCXDL0aNHjfGMjAzbOa+99poxHq5lff/+/Y3x8vJyY/zxxx+3Xeuvf/2rMf7VV1/ZzgEAAI2DOzoAAAAAPIdCBwAAAIDnUOgAAAAA8BwKHQAAAACeQ6EDAAAAwHPounYJGTt2rDFeVFRkOyc3NzdS6QCN4ptvvrEdu+mmm4zxjh072s6x+z4qLS01xrds2RImOwAA0FxxRwcAAACA51DoAAAAAPAcCh0AAAAAnkOhAwAAAMBzKHQAAAAAeA6FDgAAAADPob30JeTyyy9v6hSAS0J1dbXt2Jo1axoxEwAA0FS4owMAAADAcyh0AAAAAHgOhQ4AAAAAz6HQAQAAAOA5FDoAAAAAPMdnWZZ1sSevWLFCK1as0JEjRyRJaWlpeuaZZ5SZmSlJsixLc+fO1cqVK3Xy5En1799fy5cvV1pa2kUnVFlZqUAgoOnTp8vv9zv7agAAAAB4Rk1NjRYsWKCKigp17tzZ0VxHd3QSExO1YMECffDBB/rggw80fPhw3X777froo48kSYsWLdLixYu1bNkyFRYWKi4uTiNHjlRVVZWjpAAAAACgIRwVOqNGjdItt9yiHj16qEePHpo3b546deqk3bt3y7IsLVmyRDk5ORozZozS09O1evVqnT59WuvWrYtU/gAAAABwgXq/R+f8+fNav369qqurNWDAABUXF6u0tFQZGRnBc/x+v4YOHapdu3bZrlNTU6PKysqQAwAAAAAawnGhc/DgQXXq1El+v1+PPvqoNm3apJ/+9KcqLS2VJMXGxoacHxsbGxwzyc3NVSAQCB5JSUlOUwIAAACAEI4LnZ49e2r//v3avXu3HnvsMY0fP16HDh0Kjvt8vpDzLcu6IPZ9M2bMUEVFRfAoKSlxmhIAAAAAhGjjdEK7du101VVXSZL69eunwsJCLV26VE899ZQkqbS0VPHx8cHzy8rKLrjL831+v5/uagAAAABc1eDP0bEsSzU1NUpJSVFcXJzy8/ODY7W1tSooKNDAgQMbehkAAAAAuGiO7ujMnDlTmZmZSkpKUlVVldavX6/t27dr8+bN8vl8ys7O1vz585WamqrU1FTNnz9fHTp00H333Rep/AEAAADgAo4Kna+++koPPPCAjh8/rkAgoKuvvlqbN2/WyJEjJUnTpk3TmTNnNHHixOAHhm7ZskXR0dERSR4AAAAATHyWZVlNncT3VVZWKhAIaPr06bx3BwAAAGjBampqtGDBAlVUVKhz586O5jb4PToAAAAA0Nw47roWad/dYKqpqWniTAAAAAA0pe9qgvq8CK3ZvXTt888/50NDAQAAAASVlJQoMTHR0ZxmV+jU1dXpyy+/VHR0tHw+nyorK5WUlKSSkhLHr8vDpY29b7nY+5aLvW+52PuWi71vuS5m7y3LUlVVlRISEtSqlbN33TS7l661atXKWK117tyZJ38Lxd63XOx9y8Xet1zsfcvF3rdcP7b3gUCgXuvSjAAAAACA51DoAAAAAPCcZl/o+P1+zZ49m8/UaYHY+5aLvW+52PuWi71vudj7livSe9/smhEAAAAAQEM1+zs6AAAAAOAUhQ4AAAAAz6HQAQAAAOA5FDoAAAAAPKdZFzovvPCCUlJS1L59e/Xt21fvvfdeU6cEl+Xm5uq6665TdHS0unXrpjvuuEN/+9vfQs6xLEtz5sxRQkKCoqKiNGzYMH300UdNlDEiJTc3Vz6fT9nZ2cEYe+9dX3zxhe6//3516dJFHTp00M9+9jPt2bMnOM7ee9O5c+c0a9YspaSkKCoqSldeeaWeffZZ1dXVBc9h771hx44dGjVqlBISEuTz+fSnP/0pZPxi9rmmpkZZWVnq2rWrOnbsqNGjR+vzzz9vxK8C9RFu78+ePaunnnpKvXv3VseOHZWQkKBx48bpyy+/DFnDrb1vtoXOhg0blJ2drZycHO3bt09DhgxRZmamjh071tSpwUUFBQWaNGmSdu/erfz8fJ07d04ZGRmqrq4OnrNo0SItXrxYy5YtU2FhoeLi4jRy5EhVVVU1YeZwU2FhoVauXKmrr746JM7ee9PJkyc1aNAgtW3bVn/5y1906NAhPf/887rsssuC57D33rRw4UL9/ve/17Jly/Txxx9r0aJFeu655/S73/0ueA577w3V1dXq06ePli1bZhy/mH3Ozs7Wpk2btH79eu3cuVOnTp3SbbfdpvPnzzfWl4F6CLf3p0+f1t69e/X0009r79692rhxo4qKijR69OiQ81zbe6uZuv76661HH300JNarVy9r+vTpTZQRGkNZWZklySooKLAsy7Lq6uqsuLg4a8GCBcFzvv32WysQCFi///3vmypNuKiqqspKTU218vPzraFDh1pPPPGEZVnsvZc99dRT1uDBg23H2XvvuvXWW60HH3wwJDZmzBjr/vvvtyyLvfcqSdamTZuCjy9mn8vLy622bdta69evD57zxRdfWK1atbI2b97caLmjYX649ybvv/++Jck6evSoZVnu7n2zvKNTW1urPXv2KCMjIySekZGhXbt2NVFWaAwVFRWSpJiYGElScXGxSktLQ54Lfr9fQ4cO5bngEZMmTdKtt96qESNGhMTZe+/Ky8tTv3799Itf/ELdunXTNddcoxdffDE4zt571+DBg/Vf//VfKioqkiR9+OGH2rlzp2655RZJ7H1LcTH7vGfPHp09ezbknISEBKWnp/Nc8JiKigr5fL7gXX03976Nm4m65euvv9b58+cVGxsbEo+NjVVpaWkTZYVIsyxLU6ZM0eDBg5Weni5Jwf02PReOHj3a6DnCXevXr9fevXtVWFh4wRh7712fffaZVqxYoSlTpmjmzJl6//339fjjj8vv92vcuHHsvYc99dRTqqioUK9evdS6dWudP39e8+bN07333iuJ7/uW4mL2ubS0VO3atdM//dM/XXAOvwt6x7fffqvp06frvvvuU+fOnSW5u/fNstD5js/nC3lsWdYFMXjH5MmTdeDAAe3cufOCMZ4L3lNSUqInnnhCW7ZsUfv27W3PY++9p66uTv369dP8+fMlSddcc40++ugjrVixQuPGjQuex957z4YNG/TKK69o3bp1SktL0/79+5Wdna2EhASNHz8+eB573zLUZ595LnjH2bNndc8996iurk4vvPDCj55fn71vli9d69q1q1q3bn1B1VZWVnZB9Q9vyMrKUl5ent59910lJiYG43FxcZLEc8GD9uzZo7KyMvXt21dt2rRRmzZtVFBQoH//939XmzZtgvvL3ntPfHy8fvrTn4bEfvKTnwSbzfB9711PPvmkpk+frnvuuUe9e/fWAw88oF//+tfKzc2VxN63FBezz3FxcaqtrdXJkydtz8Gl6+zZs/rlL3+p4uJi5efnB+/mSO7ufbMsdNq1a6e+ffsqPz8/JJ6fn6+BAwc2UVaIBMuyNHnyZG3cuFHbtm1TSkpKyHhKSori4uJCngu1tbUqKCjguXCJu+mmm3Tw4EHt378/ePTr10+/+tWvtH//fl155ZXsvUcNGjTogjbyRUVFSk5OlsT3vZedPn1arVqF/urRunXrYHtp9r5luJh97tu3r9q2bRtyzvHjx/W///u/PBcucd8VOYcPH9bWrVvVpUuXkHFX995R64JGtH79eqtt27bWSy+9ZB06dMjKzs62OnbsaB05cqSpU4OLHnvsMSsQCFjbt2+3jh8/HjxOnz4dPGfBggVWIBCwNm7caB08eNC69957rfj4eKuysrIJM0ckfL/rmmWx9171/vvvW23atLHmzZtnHT582Fq7dq3VoUMH65VXXgmew9570/jx460rrrjCevPNN63i4mJr48aNVteuXa1p06YFz2HvvaGqqsrat2+ftW/fPkuStXjxYmvfvn3BzloXs8+PPvqolZiYaG3dutXau3evNXz4cKtPnz7WuXPnmurLwkUIt/dnz561Ro8ebSUmJlr79+8P+d2vpqYmuIZbe99sCx3Lsqzly5dbycnJVrt27axrr7022HIY3iHJeKxatSp4Tl1dnTV79mwrLi7O8vv91g033GAdPHiw6ZJGxPyw0GHvveuNN96w0tPTLb/fb/Xq1ctauXJlyDh7702VlZXWE088YXXv3t1q3769deWVV1o5OTkhv+Cw997w7rvvGv99Hz9+vGVZF7fPZ86csSZPnmzFxMRYUVFR1m233WYdO3asCb4aOBFu74uLi21/93v33XeDa7i19z7Lsixn94AAAAAAoHlrlu/RAQAAAICGoNABAAAA4DkUOgAAAAA8h0IHAAAAgOdQ6AAAAADwHAodAAAAAJ5DoQMAAADAcyh0AAAAAHgOhQ4AAAAAz6HQAQAAAOA5FDoAAAAAPIdCBwAAAIDn/D8hhP0I3BmRRwAAAABJRU5ErkJggg=="/>

## 신경망 구성



```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x): 
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
                         
net = Net()
print(net)
```

<pre>
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
</pre>
- `.parameters()` 



```python
params = list(net.parameters())
print(len(params))  #length 확인
print(params[0].size())
```

<pre>
10
torch.Size([6, 1, 3, 3])
</pre>
임의의 값을 넣어 forward값 확인



```python
input = torch.randn(1,1,28,28)
out = net(input)
print(out)
```

<pre>
tensor([[-0.0086, -0.0508, -0.0170,  0.0005,  0.0486,  0.0166, -0.0378, -0.0773,
         -0.1033,  0.1539]], grad_fn=<AddmmBackward0>)
</pre>
## 손실함수와 옵티마이저



```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum= 0.9)
```

## 모델 학습



- `optimizer.zero_grad` : 가중치의 그래디언트 초기화



- loss 계산



- `loss.backward()`



- `optmizer.step()` : 업데이트


- 배치수 확인



```python
total_batch = len(train_loader)
print(total_batch)
```

<pre>
469
</pre>
- 설명을 위해 `epochs=2`로 지정



```python
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader,0):
        inputs, labels = data
        
        optimizer.zero_grad()  #매개변수 초기화
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i %100 == 99:
            print("Epoch: {}, Iter: {}, Loss: {}".format(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0
```

<pre>
Epoch: 1, Iter: 100, Loss: 0.11518591153621674
Epoch: 1, Iter: 200, Loss: 0.11483499646186829
Epoch: 1, Iter: 300, Loss: 0.11444769549369813
Epoch: 1, Iter: 400, Loss: 0.11398974680900574
Epoch: 2, Iter: 100, Loss: 0.11238955092430115
Epoch: 2, Iter: 200, Loss: 0.10996400082111359
Epoch: 2, Iter: 300, Loss: 0.10370747458934784
Epoch: 2, Iter: 400, Loss: 0.08563396507501603
</pre>
## 모델의 저장 및 로드



- `torch.save`

  - `net.state_dict()`를 저장



- `torch.load`

  - `load_state_dict`로 모델을 로드



```python
PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)
```

다시 불러오기



```python
net = Net()
net.load_state_dict(torch.load(PATH))
```

<pre>
All keys matched successfully
</pre>

```python
net.parameters
```

<pre>
bound method Module.parameters of Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
</pre>
## 모델 테스트



```python
dataiter = iter(test_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images[:4]))
```

<img src="data:image/png;base64,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"/>


```python
outputs = net(images)
```


```python
_, predicted = torch.max(outputs,1)
print(predicted)
```

<pre>
tensor([7, 2, 1, 0, 4, 1, 7, 9, 6, 7, 0, 6, 4, 0, 1, 3, 4, 7, 3, 4, 7, 6, 4, 8,
        4, 0, 7, 4, 0, 1, 3, 1, 3, 6, 7, 2, 7, 1, 2, 1, 1, 7, 4, 2, 3, 5, 8, 2,
        4, 4, 6, 3, 5, 5, 6, 5, 4, 1, 4, 7, 7, 8, 4, 3, 7, 4, 6, 4, 8, 0, 7, 0,
        2, 8, 1, 7, 3, 7, 1, 7, 7, 6, 2, 7, 4, 4, 7, 3, 6, 1, 3, 6, 4, 3, 1, 4,
        1, 7, 6, 4, 6, 0, 7, 4, 9, 4, 2, 1, 7, 4, 8, 1, 3, 4, 7, 4, 4, 4, 4, 8,
        5, 4, 7, 6, 7, 9, 0, 5])
</pre>

```python
print(''.join('{}\t'.format(str(predicted[j].numpy())) for j in range(4)))
```

<pre>
7	2	1	0	
</pre>

```python
correct = 0
total =0

with torch.no_grad(): #test 중이므로
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(100*correct/total)
```

<pre>
73.84
</pre>
## GPU 설정 후 학습

- 설정 후, 모델과 데이터에 `to`로 GPU를 사용 가능한 형태로 변환 해줘야 한다.



- `.cuda()` 로도 가능



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
```

- 현재 GPU 설정



```python
torch.cuda.is_available()
```

<pre>
True
</pre>

```python
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else'cpu')
```

- 데이터 로드



```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
trainset = dsets.MNIST(root='MNIST_data/',
                      train = True,
                      download = True,
                      transform = transform)
testset = dsets.MNIST(root='MNIST_data/',
                      train = False,
                      download = True,
                      transform = transform)
train_loader = DataLoader(trainset, batch_size = 128, shuffle = True, num_workers=2)
test_loader = DataLoader(testset, batch_size = 128, shuffle = False, num_workers=2)
```

- 모델 생성



```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x): 
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
                         
net = Net()
print(net)
```

<pre>
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
</pre>

```python
net = Net().to(device)
```


```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum= 0.9)
```


```python
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader,0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()  #매개변수 초기화
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i %100 == 99:
            print("Epoch: {}, Iter: {}, Loss: {}".format(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0
```

<pre>
Epoch: 1, Iter: 100, Loss: 0.11503624880313873
Epoch: 1, Iter: 200, Loss: 0.11465224099159241
Epoch: 1, Iter: 300, Loss: 0.11420497107505799
Epoch: 1, Iter: 400, Loss: 0.11353006255626678
Epoch: 2, Iter: 100, Loss: 0.11058672618865967
Epoch: 2, Iter: 200, Loss: 0.10533232337236405
Epoch: 2, Iter: 300, Loss: 0.08904141175746917
Epoch: 2, Iter: 400, Loss: 0.06056693658232689
</pre>

```python
correct = 0
total =0

with torch.no_grad(): #test 중이므로
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(100*correct/total)
```

<pre>
81.49
</pre>
