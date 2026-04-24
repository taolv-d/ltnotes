
[megvii-research/NAFNet: The state-of-the-art image restoration model without nonlinear activation functions.](https://github.com/megvii-research/NAFNet?utm_source=chatgpt.com)

NAFNet 本质上是一个很干净的 U-Net 式图像复原网络。它的整体思路可以概括成：

```text
输入图像
  -> intro 3x3 conv
  -> 编码器多层下采样
  -> bottleneck / middle blocks
  -> 解码器多层上采样 + skip connection
  -> ending 3x3 conv
  -> 与输入做全局残差相加
  -> 输出图像
```

- `intro`：把输入 `img_channel` 映射到基础通道数 `width`
- `encoders`：每个尺度堆若干个 `NAFBlock`
- `downs`：用 `2x2, stride=2` 卷积做下采样，并把通道翻倍
- `middle_blks`：最底部再堆若干个 `NAFBlock`
- `ups`：`1x1 conv + PixelShuffle(2)` 上采样，并把通道减半
- `decoders`：上采样后先和对应 encoder 特征做逐元素相加，再过 `NAFBlock`
- `ending`：映射回输出通道
- 最后 `x = x + inp`：这是图像复原里很常见的全局残差

##  NAF block
真正有特点的是它的基本单元 `NAFBlock`。这个 block 的设计可以理解成“极简版 Transformer/Conv block”，但完全用卷积实现，没有常规激活函数。前半段是**局部混合 + 简化通道注意力**


```
输入特征 x (B, C, H, W)
  │
  ├─ 残差分支：直接跳接到最后
  │
  └─ 主分支：
       ↓
      LayerNorm（归一化）
       ↓
      1×1 Conv（通道扩展，C → C×α，通常 α=2 或 2.66）
       ↓
      3×3 Depthwise Conv（空间特征提取，通道数不变）
       ↓
      SimpleGate（关键创新）
        │
        ├─ 沿通道拆成两份：x1, x2 = chunck(2)
        └─ 输出 = x1 ⊙ x2（逐元素相乘）
       ↓
      SCA（简化通道注意力）
        │
        ├─ 全局平均池化：(1, C, 1, 1)
        ├─ 1×1 Conv（降维再升维，带激活）
        └─ Sigmoid → 再乘回输入
       ↓
      1×1 Conv（通道投影，C → 原C）
       ↓
      + 跳接（残差相加）
  │
  ↓
输出 y (B, C, H, W)
```

几个关键点：

- `SimpleGate`不是 ReLU/GELU，而是把通道一分为二后相乘：`x1 * x2`
- `sca`是 `AdaptiveAvgPool2d(1) + 1x1 conv`，比 SE 更轻
- 两个残差缩放参数 `beta` 和 `gamma` 初始化为 0，这让网络一开始更接近恒等映射，训练会更稳
- `LayerNorm2d` 是对每个空间位置沿通道维做归一化，不是 BatchNorm



## SCA（Simplified Channel Attention)

SCA 是一种简化注意力机制的方法，但是后续的算法复用的不多

## # 传统通道注意力（如 SENet）的完整流程：
```
输入特征 (B, C, H, W)
  ↓
全局平均池化 (B, C, 1, 1)
  ↓
全连接层1（降维，C → C/r，r=16）
  ↓
ReLU 激活
  ↓
全连接层2（升维，C/r → C）
  ↓
Sigmoid 激活 → 得到通道权重 (B, C, 1, 1)
  ↓
逐通道乘回原特征
```

###  SCA 的简化流程：
```
输入特征 (B, C, H, W)
  ↓
全局平均池化 (B, C, 1, 1)
  ↓
1×1 卷积（C → C，注意：**不降维**）
  ↓
Sigmoid 激活 → 得到通道权重 (B, C, 1, 1)
  ↓
逐通道乘回原特征
```

## 两者的关键区别：

| 对比项 | 传统通道注意力 | SCA（简化版） |
|--------|---------------|---------------|
| 降维操作 | 有（C→C/r→C） | **无** |
| 全连接层 | 2个 | **0个**（用1×1卷积代替） |
| 激活函数 | ReLU + Sigmoid | **仅Sigmoid** |
| 参数量 | 较多（2个FC层） | **极少（1个卷积层）** |
| 跨通道交互 | 通过降维实现 | **直接学习每个通道权重** |
