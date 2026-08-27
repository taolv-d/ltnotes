---
type: note
status: done
tags:
  - machine-learning
  - nn-block
  - conv
rating: 0
create: 2026-04-14
update: 2026-08-27
---
Conv block 通常包含三个部分：
- Conv 卷积：提特征
- BN 归一化：稳训练（Batch Normalization）
- Act 非线性：加入非线性表达能力（如ReLU）
- 可选池化：如Max Pooling, avg Pooling

Batch Normalization 对每个通道，在一个 batch 上的特征值做标准化。
1. 对每个通道单独统计均值标准差
2. 通道归一化（均值为0，标准差1 的分布），稳定训练。
也可以带上可学习的参数，例如：
$$
\begin{split}
\hat{x} &= \frac{x - \bar{x}}{\sqrt{\sigma + \varepsilon}}\\
y &= \gamma\hat{x} + \beta\\
\end{split}
$$
前半段是标准化。  
后半段 $\gamma$ 和 $\beta$ 是可学习参数，网络可以自己决定缩放平移