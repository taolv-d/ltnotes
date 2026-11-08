---
type: note
status: draft
tags:
  - mtf
rating: 0
create: 2026-11-08
update:
---
彩色相机 raw 图测试 SFR 相比黑白图像测量，会复杂一些，你要面对带有马赛克的raw图。通常常有几种处理方法
# demoniac 处理
一种方法是对 raw 图做 demoniac，但是demoniac 算法本身也会影响清晰度。例如最简单的双线性插值往往会引入模糊，造成测试结果低于真实结果。这篇文章^[https://ieeexplore.ieee.org/abstract/document/10742328] 评估了双线性插值，发现：
>**双线性插值会显著压低 SFR，且在所有测试算法中通常是最低的**；它的 SFR50 比 Bayer 基准低约 33%，SFR@Ny 只有 Bayer 的一半左右。
# raw 域直接算
另外一种方法是在稀疏 raw 图（只取某一通道）上计算。这篇文章介绍了了一种实践方案^[https://ui.adsabs.harvard.edu/link_gateway/2018JOSAA..35..442V/doi:10.1364/JOSAA.35.000442] 。文章给出两个改进点：
1. 低于畸变，直接用曲线拟合，不用做畸变校正（或图卡的补偿）。图中展示的实线对应直接直线拟合的结果，由于边缘错位，得到的结果会更模糊。
![[attachments/Pasted image 20261108204037.png|369]]
2. 第二个就是稀疏采样，如下图所示，只用某一个颜色通道采样得到ESF
![[attachments/Pasted image 20261108204635.png|446]]
作者的方案不难想到，甚至可能一些自研算法已经采取了类似方案。稀疏采样并不是完美的方案，原因是稀疏采样引起的原始数据不足，**需要更长的边缘来弥补**，边缘太长也可能造成测量区域的SFR本身有差异。
