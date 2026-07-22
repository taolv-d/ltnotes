---
type: artical
status: draft
tags:
  - machine-learning
  - backbone
  - mamba
rating: 0
create: 2026-04-25
update: 2026-06-26
url: https://arxiv.org/abs/2312.00752
---
Mamba 是一类模型家族，正统是SSM发展而来的。图像领域还有类似 ViT、Swin 这种将Transformer 改造成适用图像的 vision-mamba。

这里主要介绍 SSM 发展过来的正统mamba 跟 vision mamba

## SSM 到 mamba

这里简单记录，详细内容可以参考这篇博客，写的很细、比较全：https://blog.csdn.net/v_JULY_v/article/details/134923301

mamba 的发展可以看作是从(简化了) RNN -> SSM -> S4 -> mamba
这里回顾下这几个关键模型架构：
### RNN
RNN 网络通过隐层记录历史信息，来实现生成任务的，t 时刻的输出需要依赖t-1时刻更新隐层。他的优点是推理速度快。但前后依赖关系造成如何两个问题：
	1. 前后的依赖，造成训练时只能顺序训练，无法并行。
	2. 隐层存储的历史长度有限，一旦满了就要把最早的清楚，即容易遗忘。

### SSM
SSM是对RNN的改进，他可以并行训练，对比RNN 跟SSM如下：
**RNN的公式**：
$$ht=激活函数(W⋅h_{t−1}+U⋅x_t)$$
这里的问题是激活函数是非线性的，$h_t,h_{t-1}$不满足结合律，没办法把公式递归展开。因此不能GPU并行计算。

**SSM的公式**：
$$\begin{split}
h_t​
&=A\cdot h_{t−1}​+B\cdot x_t\\
y_t
&=C\cdot h_t
\end{split}​$$ 这里第一个公式有时序依赖，但他是全线形变换，没有非线性激活函数（非线性被放在SSM模块之外）。因此**结合律成立**，就可以做例如下面这样的展开了：
$$
\begin{split}
h_1&​=Ah_0​+B_1​x_1​\\
h_2=Ah_1+B_2x_2&=A^2h_0+A(B_1x_1)+(B_2x_2)\\
h_3=Ah_2+B_3x_3&=A^3h_0+A^2(B_1x_1)+A(B_2x_2)+(B_3x_3)
\end{split}
$$
这样就可以做并行计算，甚至是类似CNN 的样式。同时推理时又能恢复成RNN的方式高效推理

##HiPPO

SSM 解决了RNN 不能并行训练的问题，但是无法解决遗忘问题。HiPPO 可以把记忆压缩到一个矩阵中，来弥补遗忘问题。详细可以参考本节开头提到的文章。

### S4

S4 就是离散SSM+HiPPO。这里也不展开了，主要指出他的不足：即A B C 三个权重矩阵都是固定的，因此对所有输入都是一视同仁，即没有Transformer 那种注意力机制，这也是mamba 要解决的问题。

### mamba

mamba 解决注意力问题是通过 让 B C 两个矩阵不在是原来的死矩阵了，而是会随输入的$x_t$变化。这样就有了注意力机制。
这里再多说下并行训练，首先 mamba 继承了SSM ，也可以展开。mamba 并行训练流程如下：
1. 先分块：假设序列长度 L=4096，GPU 把它分成 4 个块（Block），每块 1024 个 token：
	- 注意这里都假设初始状态为0
	
	- **Block 1（GPU 核心 1）**：顺序算出 $h_1→h_{1024}$（串行递推， $h_0=0$）。
	- **Block 2（GPU 核心 2）**：顺序算出 $h_{1025}→h_{2048}$（串行递推， $h_{1024}=0$）。
	- **Block 3（GPU 核心 3）**：顺序算出 $h_{2049}→h_{3072}$（串行递推， $h_{2048}=0$）。
	- **Block 4（GPU 核心 4）**：顺序算出 $h_{3073}→h_{4096}$​（串行递推， $h_{3072}=0$）。

2. 把分块串起来，此时 $h_{1024},h_{2048},h_{3072}$ 已经算好了，此时以**Block 2**为例（其他同理）： **Block 2** 拿到真正的 $h_{1024}$​ 后，在自己内部将之前算出的 $h_{1025}^′→h_{2048}^′$​ **整体乘上一个 A 的幂次偏移量**（因为 $h_{1025}=A⋅h_{1024}+h_{1025}^′$​），这样$h_{1025}→h_{2048}$就全都算对了

# vision mamba

vision mamba 在[Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model]([[2401.09417] Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)) 中提出，简称ViM。

## ViM vs. ViT
[[ViT]]是一次性让所有图像块"两两对视"(Transformer的注意力机制，$O(n^2)$复杂度)；而ViM是让图像块排成一队，依次走过一个"有记忆的状态空间"，边走边更新记忆，最后用这个记忆代表整幅图（SSM的方式，$O(n)$复杂度）。

对 **512×512 彩色图**，Mamba 处理的总体流程如下：

| 步骤       | ViT 的做法                               | Mamba 的做法                      |
| :------- | :------------------------------------ | :----------------------------- |
| 图像分块     | 切成 16×16 的 Patch，共 1024 个             | **完全相同**（也可用其他大小）              |
| 线性投影     | 每个 Patch 映射到 D 维向量                    | 完全相同                           |
| 位置编码     | 加一个可学习的位置向量                           | 通常加上（ViM 等变体使用）                |
| **序列处理** | 所有 Patch **同时输入** Transformer，两两计算注意力 | Patch **依次输入** SSM，边走边更新隐藏状态   |
| 全局表征     | 通过 `[CLS]` token 聚合                   | 通过**最终隐藏状态**或 `[CLS]` token 聚合 |
| 复杂度      | **O(N²)**，N 是 Patch 数量（特征块两两交互）       | **O(N)**，线性增长（特征块按顺序依次交互）      |
| 规模放大     | 不容易饱和                                 | 容易饱和                           |

## ViM vs. mamba

虽然都是利用SSM，但是他们输入的数据有很大差异：
- mamba：文本，因果序列，$y_t$的输出只能看到$t$时刻及以前的信息
- ViM: 图像，非因果数据，图像没有固定的"起点"和"终点"
因此 ViM 引入了双向扫描(扩展方案有四向扫描,增加了斜向), 确保每个位置都能看到前后的信息：
>前向扫描：从左到右，[CLS] → Patch 1 → Patch 2 → ... → Patch 1024
>后向扫描：从右到左，Patch 1024 → ... → Patch 1 → [CLS]
>融合两个方向（通常是加权平均或拼接后投影）