---
type: article
status: done
tags:
  - nn-block
  - attention
rating: 0
create: 2026-04-14
update: 2026-08-30
publish: 2021-03-01
url: https://arxiv.org/pdf/2103.02907
---
Coordinate Attention（坐标注意力） 是一种**注意力模块**，比SE保留更多位置信息。
![[attachments/Pasted image 20260830180644.png]]
上图（a)是 SE 通道注意力的做法[[SE Squeeze-and-Excitation (Channel Attention)]]，常见做法是：
- 对整张特征图做全局池化
- **得到每个通道一个标量**。（这里把位置信息压缩的太狠了，没法细化到那个区域更重要）
- 再给每个通道乘一个权重
图（c）是本文的做法
- 不直接做 HxW -> 1x1 的全局池化，**而是分别沿着两个方向池化**：
    - 沿宽度池化，保留高度信息，得到 H x 1
    - 沿高度池化，保留宽度信息，得到 1 x W
- 再把这两条信息融合，最后生成两张注意力图：
    - 一张是按高度变化的权重 $a_h$
    - 一张是按宽度变化的权重 $a_w$
- 输出是：$out = x * a_h * a_w$
也就是**同时按“横向位置”和“纵向位置”去调制特征**。
**效果**：下图摘自论文，可以看到CA的注意力会更准
![[attachments/Pasted image 20260830181951.png]]
**为什么不更近一步让每个pixel都有一个注意力权重呢？**
1. **计算成本更高**  
2. **更容易过拟合或学到“碎”的东西**，注意力自由度太大了
3. **CA保留了一部分空间信息已经够用**
