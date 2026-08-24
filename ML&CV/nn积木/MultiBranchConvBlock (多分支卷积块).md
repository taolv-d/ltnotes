---
type: note
status: done
tags:
  - machine-learning
  - nn-block
  - conv
rating: 0
create: 2026-04-14
update:
---
MultiBranchConvBlock（多分支卷积块）主要是为了提高轻量模型的性能。遇到的问题是：对于轻量模型，如果想要提升性能需要加更多的卷积层来提取特征，但这损害了速度。**有没有办法让速度损失发生在训练阶段，推理时不影响呢？**

MultiBranchConvBlock 的解决办法就是：**训练的时候用各种不同的卷积提取更多特征，推理时将他们合并成一个，不影响推理的性能**。

利用了卷积、BN等操作的线性特点，训练时分开，推理时组合。下图（具体细节先不扣了）：
1. 左侧训练时候使用各种形状的卷积来提取不同特征
2. 右侧推理时把这些卷积合并成一个运算，节省计算量
![[attachments/Pasted image 20260824232723.png|580]]

