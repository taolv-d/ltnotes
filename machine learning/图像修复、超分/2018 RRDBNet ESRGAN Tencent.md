**2018年** 随论文 **ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks**

RRDBNet 网络结构：

| 层级   | 组件            | 输入→输出通道 | 说明                                           |
| ---- | ------------- | ------- | -------------------------------------------- |
| 1    | `conv_first`  | 3 → 64  | 浅层特征提取                                       |
| 2~24 | `RRDB × 23`   | 64 → 64 | 核心深层特征处理，每个 RRDB 内含 5 层密集连接 + 残差缩放 (×0.2)    |
| 25   | `conv_body`   | 64 → 64 | 后处理卷积                                        |
| 26   | `Trunk Skip`  | -       | `conv_first` 与 `conv_body` 输出相加（长跳跃连接，逐元素相加） |
| 27   | `Upsample ×2` | 64 → 64 | 每次：Conv 64→256 + PixelShuffle 还原 64 图像宽高翻倍   |
| 28   | `conv_last`   | 64 → 3  | 重建输出 RGB                                     |
[[../nn积木/RRDB|RRDB]]

网络中卷积层的特殊操作：

|模块|Conv|BN|Activation|
|---|---|---|---|
|`conv_first`|✅ 3×3|❌|✅ LReLU|
|RRDB 内各层|✅ 3×3|❌|✅ LReLU|
|`conv_body`|✅ 3×3|❌|❌|
|Upsample Conv|✅ 3×3|❌|✅ LReLU|
|`conv_last`|✅ 3×3|❌|❌|

### 📌 为什么去掉 BN（批归一化）？

这是 ESRGAN / RRDBNet 的一个重要设计选择：

- **BN 层在超分任务中容易引入伪影**：BN 会对特征图做均值和方差的归一化，这会破坏图像原有的对比度和色彩分布信息
    
- **节省显存和计算**：去掉 BN 后，网络更轻，训练也更稳定
    

### 📌 那激活函数用的是什么？

RRDBNet 全程使用 **LeakyReLU**（负斜率默认为 0.2），位置如下：

- `conv_first` 之后
    
- 每个 RRDB 内部的每个卷积层之后
    
- `conv_body` 无激活（它只是后处理卷积）
    
- PixelShuffle 前后的卷积层之后
    