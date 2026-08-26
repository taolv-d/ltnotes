---
type: note
status: done
tags:
  - machine-learning
  - nn-block
rating: 0
create: 2026-04-14
update: 2026-08-26
---
Spatial Pyramid Pooling - Fast（快速空间金字塔池化，SPPF），他是 YOLO 里一个很经典的“**扩大感受野、融合多尺度上下文**”的小模块。

**它解决什么问题** ：通过对输入特征图进行不同尺度的池化操作，生成固定长度的特征向量，从而使得网络能够处理任意尺寸的输入图像（保持输入输出特征图尺寸一致）。此外，他还可以多尺度特征融合、扩大感受野。

下图分别是SPP 跟 SPPF 示意图，他们都有相同的作用，但是SPPF计算量要少一些 
![[attachments/Pasted image 20260826223910.png]]