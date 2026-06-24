---
type: note
status: done
tags:
  - camera
  - isp
rating: 0
create: 2026-06-02
update:
---

LMMSE 全称是 Linear Minimum Mean Square-error Estimation。对应论文是：
```text
L. Zhang and X. Wu,
Color demosaicing via directional Linear Minimum Mean Square-error Estimation,
IEEE Transactions on Image Processing, 2005.
```

定位：
```text
高 ISO / 噪声友好 demosaic
目标是在插值中考虑噪声（利用局部方差/残差评估噪声水平，用噪声少的插值）
不过算法插值利用的信息比较少，例如没有斜向信息，斜向边缘表现稍差
```

LMMSE 默认会在 gamma 域插值，写回时反gamma。也可以不在 gamma 域计算


## 1. 插值G

**水平垂直色差估计**
第一步在 R/B 位置估计色差，这里算法比较简单，仅在水平垂直两个方向估计，公式大概如下：
```text
色差=当前点左右两个G通道均值 - (当前C*1/2 + 左右两个C之和*1/2）
实际就是邻域几个像素插值出来的，没有各种梯度权重
```
这一步有简单的高亮/异常保护：
```text
如果当前 CFA 明显高于局部亮度估计 Y，
    用 median 候选替换。
否则
    限制到 0..1。
```

**色差低通滤波**

LMMSE 对水平差分和垂直差分分别做一维 9-tap 低通，滤波核是近似高斯平滑：
```text
[exp(-16/8), exp(-9/8), exp(-4/8), exp(-1/8), 1, ....] / sum
```

**色差融合（降噪）、G插值**

这里的思想是：
1. 低通滤波前后的色差的差值（残差）反应噪声水平
2. 低通滤波后的值 反应真实的结构变化
3. 融合：结构变化大，那就不能太平滑

具体计算过程如下
```cpp
obs_h    // 原始水平差分估计
mean_h   // 低通后的水平差分估计

// 在 9 点窗口中统计：
mu = average(mean_h[窗口])
vx = eps + sum((mean_h - mu)^2)    // 低通后信号方差，反应结构强度
vn = eps + sum((mean_h - obs_h)^2) // 低通前后残差能量，反应噪声

// 估计值
// 估计的色差值
// vx越大->结构越强->不能过度平滑，多用滤波前的
// vn越大->噪声越高->需要降噪，多用滤波后的
xh = (obs_h * vx + mean_h * vn) / (vx + vn)
// 反应估计值的不准确性，用于后续的水平垂直融合，越大越不准（类似方差）
vh = vx * vn / (vx + vn)

// 同理得到垂直方向
xv
vv

// 水平垂直方向融合，得到最终色差。这里也是：inverse-variance fusion
// 也就是自己的不准，就给对立方向提高权重
final = (xh * vv + xv * vh) / (vh + vv)

// 有了色差，直接加回去就是G了
```

## 2. 补 R/B

1. 这里在绿色通道插值 R-G B-G。算法也非常简单，直接用左右或上下邻域的色差插值（具体用水平还是垂直，由当前 G 点在 Bayer 中相邻的是 R 还是 B 决定）：

```text
C@G = G + 0.5 * ((C-G)[left] + (C-G)[right])

或者：
C@G = G + 0.5 * ((C-G)[up] + (C-G)[down])
```

2. 在 R/B 点补另一个颜色：

```text
C@R/B = G + 0.25 * (
    (C-G)[up]
  + (C-G)[left]
  + (C-G)[right]
  + (C-G)[down]
)
```


## 3. 额外的增强

RawTherapee 的 LMMSE  加了一些可选的后处理：
1. median filter，减少彩噪、斑点
2. refinement，修正边缘纹理处的色彩，调整

权重类似：
```text
d = 1 / (1 + 颜色差异 + 结构差异)
```
也是利用了色差平滑的特性，给更平滑、色差差异小的方向更多的融合权重。
