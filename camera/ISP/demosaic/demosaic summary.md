---
type: note
status: draft
tags:
  - camera
  - isp
  - demosaic
rating: 0
create: 2026-06-02
update: 2026-06-25
---
这篇知乎文章可以用于快速理解demosaic算法（主要是**色差法**、基于**梯度计算权重**的思想）的原理：[基于边缘感知的低复杂度、高质量去马赛克算法 - 知乎](https://zhuanlan.zhihu.com/p/563755436)

经典 Bayer demosaic 的主流高质量算法，大多建立在“green 优先 + 色差平滑传播”这个框架上。

  典型流程通常是（2 3 跟 CFA 的排列密不可分）：
  1. 在 R/B 位置恢复 G，很多算法本质上是先估计 G-C 或 C-G 色差，再反推出 G；
  2. 在 R/B 位置恢复缺失的另一个颜色（R@B / B@R），通常沿对角方向传播色差；
  3. 在 G 位置恢复 R/B，通常沿水平/垂直方向传播色差。

  算法之间的主要区别不只在插值权重，还在：
  4. 差值权重怎么选：梯度、一致性、结构、高频分量的波动、噪声
  5. 候选值怎么构造：双线性、HA、ratio、高阶、低通引导、对角中间量等；
  6. 可靠性怎么评估：梯度、色差平滑性、局部方差、同质性、Nyquist/alias、对角一致性；
  7. 决策怎么做：硬选择或加权融合；
  8. 后约束怎么做：median、clip、高亮保护、artifact suppression、refinement。

demosaic 的效果对比，图片来自 [[../rawtherapee|rawtherapee]]
![[attachments/demosaic_comparison.png]]

rawtherapee 中的算法相对效果最好：[[../rawtherapee|rawtherapee]]
dcraw 中的则是相对比较老比较基础的算法 [[demosaic in dcraw]]
OpenCV 中只有简单的基础算法：[[demosaic in OpenCV]]

demoniac 算法汇总

| 方法     | 简述                                                                                                                | 链接                     |
| ------ | ----------------------------------------------------------------------------------------------------------------- | ---------------------- |
| AMaZE  | **A**lien **Ma**ps **Z**ipper **E**limination<br>最复杂的算法<br>考虑各个色差平面做梯度权重、反复修正等<br>考虑奈奎斯特频率等避免摩尔纹、拉链效应             | [[AMaZE]]              |
| linear | 线性差值                                                                                                              | [[demosaic in dcraw]]  |
| VNG    | **V**ariable **N**umber of **G**radients<br>8个方向梯度筛选                                                              | [[demosaic in dcraw]]  |
| AHD    | **A**daptive **H**omogeneity-**D**irected<br>会转到Lab空间处理，以差值对Lab色差波动小的为准                                           | [[demosaic in dcraw]]  |
| PPG    | **P**atterned **P**ixel **G**rouping<br>经典算法<br>梯度插绿，色差补红蓝                                                        | [[demosaic in dcraw]]  |
| DCB    | **D**irectional **C**olor **B**oard<br>方向图、多轮绿修正、色差插值                                                             | [[demosaic in dcraw]]  |
| DHT    | **D**ifference **H**istogram-based **T**ransform<br>用比例评价颜色差异大小，而非数值大小                                            | [[demosaic in dcraw]]  |
| EA     | **E**dge **A**ware 边缘感知<br>经典基础算法<br>先水平、垂直梯度差值G（边缘感知）<br>基于色差补红蓝                                                 | [[demosaic in OpenCV]] |
| HPHD   | **H**eterogeneity **P**rojection **H**ard **D**ecision<br>高通滤波得到方向图，异质性投影判断<br>根据方向图差值G（水平/垂直/四个方向混合）<br>色差差值 R B | [[HPHD]]               |
| IGV    | **I**terative **G**radient-based **V**ariance<br>特点是迭代<br>先计算多个方向的色差（水平、垂直等）<br>利用色差波动（如方差）作为权重来加权求和              | [[IGV]]                |
| LMMSE  | **L**ocal **M**inimum **M**ean **S**quare **E**rror<br>利用低通滤波区分噪声跟纹理，压制噪声                                         | [[LMMSE]]              |
| RCD    | **R**atio-**C**orrected **D**emosaicing<br>相对AMaZE简单很多，但效果优秀<br>比例矫正                                              | [[RCD]]                |

