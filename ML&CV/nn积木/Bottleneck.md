---
type: note
status: done
tags:
  - machine-learning
  - nn-block
rating: 0
create: 2026-04-14
update:
---
bottleneck层最初是在ResNet网络中初次提出，通过降低计算量使得网络深度可以进一步增加（也就是瓶颈层降低了单层的参数量，这样整体参数量不变的情况下，网络深度可以加深）。
瓶颈体现在中间通道数可以减少（残差链接这里通道相同才相加，通道不同就不用相加）。

![[attachments/Pasted image 20260826232341.png|542]]