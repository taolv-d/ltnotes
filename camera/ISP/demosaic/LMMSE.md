---
type: artical
status: done
tags:
  - camera
  - isp
  - demosaic
rating: 0
create: 2026-06-02
update: 2026-06-25
publish: 2005-01-01
url: https://www4.comp.polyu.edu.hk/~cslzhang/paper/LMMSEdemosaicing.pdf
---
LMMSE 全称是 Linear Minimum Mean Square-error Estimation。是一篇2005年论文提出的方法，原文是： Color demosaicing via directional Linear Minimum Mean Square-error Estimation
IPOL这篇文章介绍了算法的实现，更容易理解：[Zhang-Wu Directional LMMSE Image Demosaicking](http://www.ipol.im/pub/art/2011/g_zwld/revisions/2011-09-01/article.pdf)

该算法的主要特点是：在插值过程中**估计噪声强度**，优先用**噪声小的部分插值**。文章中实现主要是如下步骤：
1. G 插值：分别在**水平、竖直**两个方向估计噪声强度（利用LMMSE方法，先拉普拉斯插值得到带噪信号，然后低通滤波来估计噪声的强度），以及G通道插值结果
2. G 插值融合：融合水平、竖直两个方向的插值，噪声高的权重小
3. R B 插值：有完整G之后，在色差平面用邻域补 R B。

因此，这个算法 **高 ISO / 噪声友好**，不过假设噪声都是高频。不过这个方法天然也会向多用更平坦的区域插值，也符合色差平稳的特性。不过没有斜向信息，斜向边缘表现稍差。
>rawrherapee 中的实现可以选择做 gamma 变换后插值，也可选做中值滤波、边缘增强等后处理

# 算法步骤
## 1. 插值G

### **水平垂直色差估计**

这里以GR行为例（其他方向/颜色是一样的），第一步先用拉普拉斯插值，估计带噪的G：
$$
\hat{G}_n=\frac{1}{2}(G_{n-1}+G_{n+1})-\frac{1}{4}(R_{n-2}-2R_{n}+R_{n+2})
$$
R的估计类似
$$
\hat{R}_n=\frac{1}{2}(R_{n-1}+R_{n+1})-\frac{1}{4}(G_{n-2}-2G_{n}+G_{n+2})
$$
有了这两个就有了完整 GR 色差了。

>rawrherapee 中这一步有简单的高亮/异常保护：
>1. 如果当前 CFA 明显高于局部亮度估计 Y，用 median 候选替换；
>2. 否则，限制到 0.1。


### **色差低通滤波**

这里对水平差分和垂直差分分别做一维 9-tap 低通，滤波核是近似高斯平滑（不同是实现由差异）
IPOL 类似这个：
```
[4/128, 9/128, 15/128, 23/128, 26/128, ....]
```

rawrherapee类似下面的：
```text
[exp(-16/8), exp(-9/8), exp(-4/8), exp(-1/8), 1, ....] / sum
```

### **色差融合（降噪）、G插值**

**IPOL 介绍了融合权重为什么这样定：**
经过前面的计算，我们已经得到水平、竖直两个色差图 $\Phi_{g,r},\Phi_{g,b}$ ，对于一个带计算的位置，他的水平、竖直方向估计值分别为$h,v$, 对应真实估计误差（无法计算）为$\varepsilon_h,\varepsilon_v$ ，利用低通滤波残差计算得到的噪声强度估计（方差）为$\sigma_h^2,\sigma_v^2$。
融合的结果就是：
$$
\omega=(1-\lambda)h+\lambda v
$$
这里希望找到一个$\lambda$让融合后的误差最小，即$(1-\lambda)\varepsilon_h+\lambda \varepsilon_v$ 最小。对其求导并令倒数为0有：
$$
\begin{split}
0
&=\frac{\partial}{\partial\lambda}\mathbb{E}[(1-\lambda)\varepsilon_h+\lambda\varepsilon_v]^2\\
&=2\mathbb{E}[-(1-\lambda)\varepsilon_h^2+\lambda\varepsilon_v^2+(1-2\lambda)\varepsilon_h\varepsilon_v]\\
&\approx2(1-\lambda)\sigma_h^2+2\lambda\sigma_v^2
\end{split}
$$
由此可以得到：$\lambda=\frac{\sigma_h^2}{\sigma_h^2+\sigma_v^2}$

**rawtherapee 的思想是：**
1. 低通滤波前后的色差的差值（残差）反应噪声水平
2. 低通滤波后的值 反应真实的结构变化
3. 融合：结构变化大，那就不能太平滑

伪代码如下
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

# 效果对比

前面也提到，该算法斜向信息使用的不多，因此在处理斜向边缘时表现会下降，下图摘自IPOL的附图，更多的对比可以到IPOL文章中看。（右下为LMMSE）
![[attachments/Pasted image 20260625192526.png|613]]