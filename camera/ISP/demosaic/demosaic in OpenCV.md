 OpenCV 4.x demosaic 算法主要有以下三类：
  - Bilinear
  - VNG: 见 [[demosaic in dcraw]]
  - EA: Edge-Aware

## Bilinear

这是最基础的版本，逻辑最简单：
```
G@R/B: 用上下左右的 green 做线性插值
R@G / B@G: 用左右或上下的同色差分做线性插值
R@B / B@R: 用对角邻居做线性插值
```

## EA
这里的算法逻辑也是比较简答的：
```
G@R/B：
     计算上下平均和左右平均两个 green 候选
     计算垂直/水平梯度
     选梯度较小方向的 green

R@B / B@R: 用基础邻域平均补
R@G / B@G: 用基础邻域平均补
```
