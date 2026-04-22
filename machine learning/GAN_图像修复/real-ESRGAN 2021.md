## 文章提出模拟真实图像退化来生成训练数据：
![[attachments/Pasted image 20260417163334.png]]
## 两阶段训练策略

**第一阶段：训练 Real-ESRNet（PSNR 导向）** 先训练一个稳定性高、保真度强的模型，为后续的 GAN 训练打好基础。

**第二阶段：训练 Real-ESRGAN（GAN 导向）**

- **目标**：在稳定保真的基础上，通过对抗训练来增强图像的感知细节和真实感。
    
- **初始化**：使用第一阶段训练好的 **Real-ESRNet** 模型作为生成器的起点。
    
- **损失函数与权重**：组合使用 **L1 Loss**、**感知损失** 和 **GAN Loss**，论文给出的权重比例为 **{1, 1, 0.1}** 。

## 网络架构
生成器[[RRDBNet]]

## 修改成1：1时出现格子纹问题：

格子纹的问题是不是最好换网络来解决？

是的，**如果格子纹是细密、规则、固定周期的 4-phase/网格伪影，换网络通常是更根本的解决方案**。

当前 RRDBNet 在 scale=1 下会用 pixel_unshuffle(scale=4)，这等于把图像按 4×4 相位拆到通道里再处理。优点是省显存、感受野大；缺点是很容易产生**相位不一致**，在平坦区域就会表现成格子纹/棋盘纹。

**优先建议**

- **换成不使用 pixel_unshuffle 的 x1 图像复原网络**，比如：
    - NAFNet：去噪/低照复原很合适，平坦区域通常更稳
    - Restormer：质量强，但训练/推理成本更高
    - U-Net / NAFNet-like U-Net：更简单稳定
    - SwinIR x1 denoising/restoration：也可试，但窗口机制也可能有轻微块感
- 对你这个任务，我最推荐先试 **NAFNet**，因为它本来就更像 denoise/restoration，而不是 SR/GAN 纹理合成。

**不换网络的缓解办法**

- 降低/去掉 LPIPS、Perceptual，提高 L1/Charbonnier
- 加平坦区约束，比如 flat-region loss / TV loss / low-frequency consistency
- 加 4-phase consistency loss，专门约束 pixel_unshuffle 的四相位输出别错位
- 推理时确认不是 tile seam：用更大 tile 或整图跑一张小 crop 对比

**我的判断**

- 如果只是轻微格子纹，可以靠 loss 缓解。
- 如果格子纹在平坦区域稳定出现，尤其周期像 4/8/16 像素，**继续在 RRDBNet x1 上调 loss 会很费劲**。
- 更值得投入的是：用 NAFNet 或类似 x1 restoration 网络，沿用你的 Topaz/GT 配对数据重新训练一个基线。