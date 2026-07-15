---
type: artical
status: done
tags:
  - ML
  - IR
rating: 0
create: 2026-04-23
update: 2026-07-15
url: https://arxiv.org/abs/2204.04676
publish: 2022
---
原文：Simple Baselines for Image Restoration
github 代码：[GitHub - megvii-research/NAFNet: The state-of-the-art image restoration model without nonlinear activation functions. · GitHub](https://github.com/megvii-research/NAFNet)

# 解决的问题

在 NAFNet 之前，各种SOTA模型已经把网络架构设计的过于复杂：
- **块间复杂度**：多阶段网络、多尺度特征融合![[attachments/Pasted image 20260715213816.png]]
- **块内复杂度**：复杂的注意力机制、门控单元

因此本文通过实验来验证这些复杂度是否必要

# 简单网络也能效果很好

本文主要发现三点可简化部分：

- 简单模型效果就很好：简单的单阶段UNet + 基础组件（LayerNorm、GELU、通道注意力）
- 激活函数引入的非线性可以用门控单元替代（SimpleGate，将特征图在通道维度切两半，直接逐元素相乘）
- 通道注意力（CA）里的激活函数也是多余的

# NAFBlock

NAFNet 是 **Nonlinear Activation Free Network，无非线性激活网络** 的缩写，故名思义，它没有非线形的激活函数。NAFBlock 是他的基本单元。
下图（d）是NAFBlock 的基本结构，完全用卷积实现，没有常规激活函数。其中SimpleGate 引入非线形，SCA 引入通道注意力。
![[attachments/Pasted image 20260715214828.png]]

## Simple Gate

下图（c）是Simple Gate的示意图，非常简单，首先将输入在通道纬度拆成两部分（图中未画出），然后这两部分逐元素相乘。
可以想象，其中一部分会变成还是提取的特征，另外一部分变成这些特征重要性的权重。这个思想在后续很多工作中也有复用。
![[attachments/Pasted image 20260715215339.png|672]]

## SCA

SCA 是对 通道注意力CA的简化。
上图（a）是CA，可以看到计算注意力部分经过了：
```python
池化 -> 1*1 conv -> ReLU -> 1*1 conv -> sigmoid
```
（b）是SCA，它简单多了
```python
池化 -> 1*1 conv
```