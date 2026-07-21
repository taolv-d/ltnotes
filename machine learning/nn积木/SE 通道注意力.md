---
type: artical
status: draft
tags:
  - machine-learning
  - nn-block
rating: 0
create: 2026-04-21
publish: 2019-01-01
url: https://arxiv.org/pdf/1709.01507
update: 2026-07-21
---

SE 全称是 **Squeeze-and-Excitation**: 传统的CNN擅长通过卷积操作捕捉空间模式，但它们隐含地处理通道依赖性，限制了它们根据任务的重要性自适应地强调或抑制特征的能力。SE 是一个非常简单的改进，实现了：
- 网络自己判断哪些通道重要
- 重要通道放大
- 不重要通道压低

![](./attachments/Pasted%20image%2020260425174458.png)

SE 执行过程(输入为$H*W*C$)：
1. **挤压**：Global Average Pooling
   H x W x C → 1 x 1 x C

2. **激励**： 
   - FC / 1x1 conv 降维：C → C/r
   - 激活
   - FC / 1x1 conv 升维：C/r → C
   - sigmoid / hard sigmoid：得到每个通道的权重

   数学公式表示为：
   $$s = F_{ex}(z,W) = \sigma(W_2 \delta(W_1 z))$$
   其中：
   $\sigma$ 为sigmod函数，$\delta$ 为ReLU激活
   W₁、W₂是学习到的参数矩阵。W₁将维度按因子r（通常为16）减小; W₂扩展回原始通道维度C。

3. 原特征逐通道乘权重
