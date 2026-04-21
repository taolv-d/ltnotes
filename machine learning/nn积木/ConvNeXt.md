
**ConvNeXt** 是一种现代 CNN 网络架构，可以理解成“用纯卷积方式吸收 Transformer/ViT 设计经验后的卷积网络”。用于现代高性能 CNN

它的一个典型 block 大概是：

```
Input
→ 7x7 Depthwise Conv，大感受野
→ LayerNorm，LN Transformer/ViT风格，CNN喜欢BN
→ 1x1 Conv / Linear，通道扩展 4 倍
→ GELU，(Linear → GELU → Linear) 这也是transformer的风格，包括GELU也是Transformer喜欢用
→ 1x1 Conv / Linear，通道压回
→ LayerScale， 给残差分支加一个可学习的缩放系数：y = x + γ * F(x)，  γ 按通道参数
→ Residual Add
```

ConvNeXt Block 更像

```
Depthwise Conv 做空间混合
→ 1x1 Conv 做通道混合
→ GELU 激活
→ 残差连接
```

具体展开：

```
输入: H x W x C

1. 7x7 Depthwise Conv
   输出: H x W x C

2. LayerNorm
   输出: H x W x C

3. 1x1 Conv / Linear expand
   C -> 4C

4. GELU

5. 1x1 Conv / Linear project
   4C -> C

6. Residual Add
   输出: H x W x C
```

