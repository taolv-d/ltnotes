---
type: artical
status: done
tags:
  - 
rating: 0
create: 2026-07-14
update:
publish: 2004-01-01
url: https://ieeexplore.ieee.org/document/1326587
---
原文：High-Quality Linear Interpolation For Demosaicing Of Bayer-Patterned Color Images

本文是一个非常简单的去马赛克算法，只需要**加减乘除**，不需要任何分支判断就能进行。因此也是一个**可微**的demoniac算法。
github 上有实现的代码，只要100多行：https://github.com/Berezniker/CV_Bayer

本文的原理很简单：它发现，在RGB域，边缘、纹理等梯度信息在三个颜色域是高度相关的。因此在插值时不仅要看相同颜色的邻域，还要考虑其他颜色的梯度。
具体做法分为两步：
1. **基础线性差值（邻域均值）**
2. **加上其他颜色通道的梯度修正（$\alpha$混合，混合系数作者通过统计大量图像得到）**
上面两步完全可以用一个设计好的卷积核实现。因此可以无缝应用到CNN网络中（有些NN ISP 就是这样做的[[../../NN_ISP/2026 CtrlISP Hikvision|2026 CtrlISP Hikvision]]）。下图摘自原文的卷积核：
注意图中卷积核上的数字要对应原文的公式，这里主要了解它的思想，实际工程实践中没有见到直接用这个算法的，这里记录主要是有些NN ISP 为了方便反向传播用到了它。实际实现可以参考原文。

![[attachments/Pasted image 20260714231132.png|504]]


