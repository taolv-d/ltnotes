---
type: note
status: done
tags:
  - machine-learning
  - nn-block
rating: 0
create: 2026-04-21
update: 2026-08-19
---
depthwise separable convolution 深度可分离卷积，最早在一篇名为 “Rigid-motion scattering for image classification”的博士学位论文中。
Xception和MobileNet 是两个使用DSC的著名模型。DSC也是在这之后被大家熟知。
# DSC
下图是DSC的网络结构[^1]：
- **depthwise conv**：  每个输入通道各自用**一个**卷积核做空间卷积。
- **pointwise conv**：1 x 1 卷积，把这些通道线性组合起来，完成通道间的信息融合。
![[attachments/Pasted image 20260819221429.png|497]]
# 计算量
下图[^1]是DSC与普通卷积的对比，我们借助此图分析下他们计算量（忽略图中M、N、k）的差异：
假设输入为：$D\cdot D\cdot C_i$，输出为$D\cdot D\cdot C_o$
**普通卷积**，需要$C_o$个$D_k\cdot D_k\cdot C_i$的卷积核，则计算量为：
- 对于输入一层的一个点，需要计算 $D_k\cdot D_k$ 次运算。
- 对于一层需要计算 $D\cdot D\cdot D_k\cdot D_k$
- 对于完整 $C_i$ 个通道，需要计算 $D\cdot D\cdot D_k\cdot D_k \cdot C_i$
- $C_o$ 个卷积核全算完就是：$D\cdot D\cdot D_k\cdot D_k \cdot C_i \cdot C_o$

**DSC** 第一步，需要 $C_i$ 个$D_k\cdot D_k\cdot 1$的卷积核
- 一层的计算同样是  $D\cdot D\cdot D_k\cdot D_k$
- 每个卷积核只有一层。且只与输入的某一层运算：$D\cdot D\cdot D_k\cdot D_k\cdot C_i$
**DSC** 第二步，需要1个 $1\cdot 1\cdot C_o$ 的卷积核，且输入为一个$D\cdot D\cdot C_i$
- 直接带入普通卷积的公式为：$D\cdot D\cdot C_i \cdot C_o$
如果忽略第二部的计算量，DSC 的计算量是普通卷积的$1/C_o$
![[attachments/Pasted image 20260819221525.png|607]]
# 参数量

**前面的分析可见 DSC 的参数量少了很多，那么他的表达能力会下降吗？**
- 参数量减少，表达能力理论上限下降，但**有效表达未必下降**
- **标准卷积的问题**：**把空间信息和通道信息绑在一起**，很多权重是相互抵消或冗余的
- **深度可分离的瓶颈**：**通道间信息交互不足**，全靠最后的 1x1 卷积去融合。

**针对DSC的瓶颈也有很多改进：**
- **1x1卷积改不了，那就让输入变多，让特征在高维空间分的更开**：在 Depthwise 卷积之前，先用 1x1 卷积将通道数扩张 6 倍（如 MobileNetV2 的 Inverted Residual）。
- **残差连接（Residual）**：MobileNetV2 引入了线性瓶颈的残差结构。如果 Depthwise 卷积没能提取出有用的空间特征，残差连接可以直接将输入特征“抄”到输出，保证底层信息不丢失。
- **更有效的激活函数**：ReLU 在低维空间会破坏信息，但在高维空间表现良好。深度可分离卷积配合高维扩张，恰好避开了 ReLU 的信息破坏问题。

**实际表现**
- 在**大模型（如 ResNet50 换成深度可分离）**上，精度可能会掉 1-2 个点，但参数量减少 80%。
- 在**小模型**上，**深度可分离卷积（如 MobileNet）的精度往往高于同参数量的标准卷积** [^google]。因为标准卷积参数量虽多，但大部分权重在训练中“死掉”了（稀疏化），而深度可分离卷积的每一层权重都被高效利用，相当于用更少的参数做了更有效的学习。


# 参考
[^1]: 本文讲解了 MobileNet 中的DSC，以及后续的改进等。笔记中的图片均来自本文：[[https://zhuanlan.zhihu.com/p/166736637]]
[^google]:谷歌团队在**2017年提出MobileNet的原始论文中**，就报告了其与VGG16的对比。虽然准确率**小幅降低0.9%**，但**参数量仅为VGG16的1/32**。这也印证了用极少的参数代价换取相当的精度是可行的。



