---
type: artical
status: done
tags:
  - ML
  - IR
  - gan
rating: 0
create: 2026-04-17
publish: 2021-01-01
url: https://arxiv.org/pdf/2107.10833
update: 2026-07-21
---

原文：Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

本文的核心创新在于模拟图像退化，利用生成数据训练出效果经验的模型。

# 数据生成

![[attachments/Pasted image 20260417163334.png]]

核心创新在于 Real-ESRGAN 精密的退化模拟过程。传统方法应用一个简单的序列：模糊 → 下采样 → 噪声 → JPEG 压缩。

退化组件包括：
- 模糊核：除了标准高斯核，模型还包含广义高斯核和平台形核，以及 2D sinc 滤波器以模拟振铃伪影。
- 调整大小操作：在区域、双线性和双三次插值之间随机选择。
- 噪声模型：高斯噪声（彩色和灰色）和泊松噪声，用于模拟传感器特性。
- JPEG 压缩：以不同的质量因子应用。

# 网络架构

网络架构复用了ESRGAN 的模型，做了一些改进：[[../image super-resolution/2018 RRDBNet ESRGAN Tencent]]

**生成器增强:**
- 像素解洗牌操作：对于 ×1 和 ×2 超分辨率，空间维度减小而通道维度增加，提高了计算效率。
- 多尺度支持：架构适应不同的上采样因子（×1、×2、×4）。
![](https://paper-assets.alphaxiv.org/figures/2107.10833v2/img-3.jpeg)

**判别器重新设计:**
- U-Net 判别器：用具有跳跃连接的 U-Net 架构替换了原始的 VGG 风格判别器。
- 谱归一化：应用于稳定训练并防止在复杂退化条件下发生模式崩溃。
![](https://paper-assets.alphaxiv.org/figures/2107.10833v2/img-5.jpeg)

# 两阶段训练策略

**第一阶段：训练 Real-ESRNet（PSNR 导向）** 先训练一个稳定性高、保真度强的模型，为后续的 GAN 训练打好基础。

**第二阶段：训练 Real-ESRGAN（GAN 导向）**

- **目标**：在稳定保真的基础上，通过对抗训练来增强图像的感知细节和真实感。
    
- **初始化**：使用第一阶段训练好的 **Real-ESRNet** 模型作为生成器的起点。
    
- **损失函数与权重**：组合使用 **L1 Loss**、**感知损失** 和 **GAN Loss**，论文给出的权重比例为 **{1, 1, 0.1}** 。


# 格子纹问题（改为1：1恢复）

格子纹的问题是不是最好换网络来解决？
![[attachments/Pasted image 20260423095223.png]]

是的，**如果格子纹是细密、规则、固定周期的 4-phase/网格伪影，换网络通常是更根本的解决方案**。

当前 RRDBNet 在 scale=1 下会用 pixel_unshuffle(scale=4)，这等于把图像按 4×4 相位拆到通道里再处理。优点是省显存、感受野大；缺点是很容易产生**相位不一致**，在平坦区域就会表现成格子纹/棋盘纹。

## 优先建议

1. 换网络：**避免 pixel_unshuffle**，比如：NAFNet
2. 上针对网格的loss(不推荐)
  - 降低/去掉 LPIPS、Perceptual，提高 L1/Charbonnier
  - 加平坦区约束，比如 flat-region loss / TV loss / low-frequency consistency
  - 加 4-phase consistency loss，专门约束 pixel_unshuffle 的四相位输出别错位
  - 推理时确认不是 tile seam：用更大 tile 或整图跑一张小 crop 对比