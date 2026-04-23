NAFNet 在这份代码里，本质上是一个很干净的 U-Net 式图像复原网络，核心实现就在 [NAFNet_arch.py](/home/lvtao/code/NAFNet/NAFNet-main/basicsr/models/archs/NAFNet_arch.py:27)。它的整体思路可以概括成：

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

对应代码在 [NAFNet_arch.py:83](/home/lvtao/code/NAFNet/NAFNet-main/basicsr/models/archs/NAFNet_arch.py:83) 到 [NAFNet_arch.py:155](/home/lvtao/code/NAFNet/NAFNet-main/basicsr/models/archs/NAFNet_arch.py:155) 很直接：

- `intro`：把输入 `img_channel` 映射到基础通道数 `width`
- `encoders`：每个尺度堆若干个 `NAFBlock`
- `downs`：用 `2x2, stride=2` 卷积做下采样，并把通道翻倍
- `middle_blks`：最底部再堆若干个 `NAFBlock`
- `ups`：`1x1 conv + PixelShuffle(2)` 上采样，并把通道减半
- `decoders`：上采样后先和对应 encoder 特征做逐元素相加，再过 `NAFBlock`
- `ending`：映射回输出通道
- 最后 `x = x + inp`：这是图像复原里很常见的全局残差

如果看默认的 SIDD 配置 [NAFNet-width64.yml](/home/lvtao/code/NAFNet/NAFNet-main/options/train/SIDD/NAFNet-width64.yml:45)，参数是：

- `width: 64`
- `enc_blk_nums: [2, 2, 4, 8]`
- `middle_blk_num: 12`
- `dec_blk_nums: [2, 2, 2, 2]`

这意味着特征尺度大致是：

```text
3
-> 64
-> 64  (2个 NAFBlock)
-> 128 (down)
-> 128 (2个 NAFBlock)
-> 256 (down)
-> 256 (4个 NAFBlock)
-> 512 (down)
-> 512 (8个 NAFBlock)
-> 1024 (down)
-> 1024 (12个 NAFBlock, middle)
-> 512 (up + skip + 2个 NAFBlock)
-> 256 (up + skip + 2个 NAFBlock)
-> 128 (up + skip + 2个 NAFBlock)
-> 64  (up + skip + 2个 NAFBlock)
-> 3
```

真正有特点的是它的基本单元 `NAFBlock`，定义在 [NAFNet_arch.py:27](/home/lvtao/code/NAFNet/NAFNet-main/basicsr/models/archs/NAFNet_arch.py:27)。这个 block 的设计可以理解成“极简版 Transformer/Conv block”，但完全用卷积实现，没有常规激活函数。前半段是局部混合 + 简化通道注意力：

```text
LN
-> 1x1 conv            # 通道扩张，c -> DW_Expand*c
-> 3x3 depthwise conv  # 空间建模
-> SimpleGate          # 通道一分为二再逐元素相乘
-> SCA                 # 简化通道注意力
-> 1x1 conv            # 投影回 c
-> 残差1（乘 beta）
```

后半段是一个简化 FFN：

```text
LN
-> 1x1 conv            # FFN 扩张，c -> FFN_Expand*c
-> SimpleGate
-> 1x1 conv            # 回到 c
-> 残差2（乘 gamma）
```

几个关键点：

- `SimpleGate` 在 [NAFNet_arch.py:22](/home/lvtao/code/NAFNet/NAFNet-main/basicsr/models/archs/NAFNet_arch.py:22)，不是 ReLU/GELU，而是把通道一分为二后相乘：`x1 * x2`
- `sca` 在 [NAFNet_arch.py:36](/home/lvtao/code/NAFNet/NAFNet-main/basicsr/models/archs/NAFNet_arch.py:36)，是 `AdaptiveAvgPool2d(1) + 1x1 conv`，比 SE 更轻
- 两个残差缩放参数 `beta` 和 `gamma` 在 [NAFNet_arch.py:56](/home/lvtao/code/NAFNet/NAFNet-main/basicsr/models/archs/NAFNet_arch.py:56) 初始化为 0，这让网络一开始更接近恒等映射，训练会更稳
- `LayerNorm2d` 在 [arch_util.py:291](/home/lvtao/code/NAFNet/NAFNet-main/basicsr/models/archs/arch_util.py:291)，它是对每个空间位置沿通道维做归一化，不是 BatchNorm

所以 NAFNet 这个名字里的 “NAF” 通常可以理解为 “非线性激活被极简化了”。你可以对比一下仓库里的 baseline 实现 [Baseline_arch.py](/home/lvtao/code/NAFNet/NAFNet-main/basicsr/models/archs/Baseline_arch.py:22)：

- Baseline 用的是 `GELU + SE`
- NAFBlock 用的是 `SimpleGate + Simplified Channel Attention`

也就是说，NAFNet 的创新不在“更复杂”，而在“更少组件但效果很强”。

再补两个代码层面的细节：

- 输入尺寸会先 pad 到 `2 ** 编码层数` 的倍数，见 [NAFNet_arch.py:157](/home/lvtao/code/NAFNet/NAFNet-main/basicsr/models/archs/NAFNet_arch.py:157)，这样多次下采样不会出尺寸问题
- `NAFNetLocal` 在 [NAFNet_arch.py:164](/home/lvtao/code/NAFNet/NAFNet-main/basicsr/models/archs/NAFNet_arch.py:164)，会把全局池化替换成更适合大图推理的局部实现，相关逻辑在 [local_arch.py](/home/lvtao/code/NAFNet/NAFNet-main/basicsr/models/archs/local_arch.py:78)
