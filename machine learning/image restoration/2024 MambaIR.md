---
type: artical
status: done
tags:
  - ML
  - IR
  - mamba
rating: 0
create: 2026-04-23
publish: 2024
url: https://arxiv.org/abs/2402.15648
update: 2026-07-03
---
MambaIR 这篇文章的目的就是将 Mamba 的思想用到图像超分中去，这样就继承了Mamba的高效。Mamba 可以参考这个文章[[../backbone/Mamba|Mamba]]

# V1

Mamba 主要用于图像分类的，因此专注于 high level 特征。IR 任务需要更多关注 low level 特征。因此向 ViT ViM 这些都无法直接应用于 IR 任务。本文作者针对这些问题做了两个主要改进：
1. 再SSM后引入卷积，对抗Mamba 序列化引入的局部像素遗忘
2. 引入通道注意力（CA）来避免冗余

## 网络结构

![[attachments/Pasted image 20260703155024.png]]

整体遵循浅层特征提取、深层特征提取、图像重建的逻辑。同时通过长连接将最浅层特征与最深层特征相加，强制重建时结合低级特征跟高级语义信息。

## 卷积的引入
原文图3a 的对比可以看到 Mamba 的顺序扫描会更关注四邻域的信息，而斜对角的块在距离上更远，但在IR中他们也很重要。因此VSSM后有一个卷积，避免这些信息被遗忘（原文：To this end, we introduce an additional local convolution after VSSM to help restore the neighborhood similarity）

![[attachments/Pasted image 20260703160353.png|279]]

## 通道注意力

这里跟SE[[../nn积木/SE 通道注意力|SE 通道注意力]] 思想一致。原文没有展开介绍。

## 感受野差异

![[attachments/Pasted image 20260703161138.png]]
这里可以更直观的理解：
1. **CNN方法**: 感受野实际靠的是下采样深度，但是超分要关注细节，因此矛盾了，所以感受野难以提升。
2. **Swin等**: 虽然用了Transformer，但为了加速融合了CNN下采样的思想，因而也继承了感受野的问题
3. **MambaIR**: 首先mamba本身解决速度不是通过CNN那种下采样方式，因此不会因为加速而难以聚焦更大区域的低级特征。同时本文引入的CNN方式仅用于克服序列扫描的问题，因此不会限制感受野
MambaIR的效果可以到论文去对比。实测（使用相同的数据微调后对比）效果要比早期的SOTA（Real-ESRGAN、NAFNet）进步不少。

# V2

v2 针对v1的改进，无论速度还是效果都更好：[[2411.15269] MambaIRv2: Attentive State Space Restoration](https://arxiv.org/abs/2411.15269)

主要改进有两点：
1. mamba 的SSM 扫描机制在图像中有冗余。因此引入了 类似注意力的全局感知，因此扫描时**只需要一个方向**，大大提升了效率
2. mamba 的长距离衰减对于图像就不友好（图像希望根据相似度而非位置远近）。因此引入了语义重排，让相似的块更近。

关于V2这些改进的具体实现这里先不深入

