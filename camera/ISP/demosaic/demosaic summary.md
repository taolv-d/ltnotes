这篇知乎文章可以用于快速理解demosaic算法（主要是色差法、基于梯度计算权重的思想）的原理：[基于边缘感知的低复杂度、高质量去马赛克算法 - 知乎](https://zhuanlan.zhihu.com/p/563755436)

  
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

各个算法横向对比：
[[../../../TODO|TODO]]