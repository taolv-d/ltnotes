---
type: note
status: draft
tags:
  - camera
  - isp
  - demosaic
rating: 0
create: 2026-06-02
update:
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

| 方法     | 简述                                                   | 链接                     |
| ------ | ---------------------------------------------------- | ---------------------- |
| AMaZE  | 最复杂的算法<br>考虑各个色差平面做梯度权重、反复修正等<br>考虑奈奎斯特频率等避免摩尔纹、拉链效应 | [[AMaZE]]              |
| linear | 线性差值                                                 | [[demosaic in dcraw]]  |
| VNG    | 8个方向梯度筛选                                             |                        |
| PPG    | 经典算法<br>梯度插绿，色差补红蓝                                   |                        |
| AHD    | 会转到Lab空间处理，以差值对Lab色差波动小的为准                           |                        |
| DCB    | 方向图、多轮绿修正、色差插值                                       |                        |
| DHT    | 用比例评价颜色差异大小，而非数值大小                                   |                        |
| EA     | 经典基础算法<br>先水平、垂直梯度差值G<br>基于色差补红蓝                     | [[demosaic in OpenCV]] |
| HPHD   | 高通滤波得到方向图<br>根据方向图差值G（水平/垂直/四个方向混合）<br>色差差值 R B      | [[HPHD]]               |
| IGV    | 先计算多个方向的色差（水平、垂直等）<br>利用色差波动（如方差）作为权重来加权求和           | [[IGV]]                |
| LMMSE  | 利用低通滤波区分噪声跟纹理，压制噪声                                   | [[LMMSE]]              |
| RCD    | 相对AMaZE简单很多，但效果优秀                                    | [[RCD]]                |
[[../../../TODO|TODO]]：
- **PPG**：“梯度插绿，色差补红蓝”这个总结过于简化。它的全称是**Patterned Pixel Grouping**，核心是先对绿色像素做自适应插值，然后根据色差恒定准则来恢复红色和蓝色。但更关键的是，它通过“像素分组”的策略来估算图像噪声，并在插值中加以考虑，这才是PPG区别于简单算法的核心，而不仅仅是“经典”和“梯度插绿”。
    
- **RCD**：“相对AMaZE简单很多，但效果优秀”只说对了一半。它的核心在于 **Ratio-Corrected（比例校正）** ，这是其命名的由来，也是算法的精髓。此外，RCD在darktable中的一大特色是**可以与VNG4结合进行“双重重构”**，分别处理细节区和平坦区，而这在表格中完全没有体现。
    
- **EA (Edge-Aware)**：“先水平、垂直梯度差值G，基于色差补红蓝”确实描述了其基本流程，但“经典基础算法”的定位可能不够突出。它的核心就在于“Edge-Aware”（边缘感知），插值时引入了边缘方向检测，对后续算法有启发意义。
    
- **HPHD (Heterogeneity-Projection Hard-Decision)**：表格中的描述“高通滤波得到方向图...色差差值R B”基本正确。但它的一个关键特点是引入了**异质性投影（Heterogeneity-Projection）** 的概念来判断边缘方向，这是它名字的来源，也是区别于其他梯度算法的核心。
    
- **IGV (Iterative Gradient-based Variance)**：表格的描述“计算多个方向色差...用色差方差作为权重”是准确的。但它最核心的特色在于 **Iterative（迭代）** ，即它会通过多次迭代，不断用上一步的结果来优化当前步的插值，直到收敛。

