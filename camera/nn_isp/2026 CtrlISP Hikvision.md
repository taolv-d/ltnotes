---
type: artical
status: done
tags:
  - 
rating: 0
create: 2026-07-15
update:
publish: 2026-01-01
url: https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Zhang_CtrlISP_Rescuing_Low-Light_CVPRF_2026_supplemental.pdf
---
原文：CtrlISP: Rescuing Low-Light RAW Images via Controllable Neural ISP

# 解决问题

本文是一篇针对低光场景的ISP，它主要解决的问题是：
1. 低光噪声大
2. 低光颜色恢复难

# 网络架构

![[attachments/Pasted image 20260714232324.png]]
上图是本文的网络架构：
1. **DenoiseNet**: 这部分是一个raw域降噪的网络，采用Unet架构，但引入了 1. **位置/ISO编码**，2. **注意力机制**
	1. 位置/ISO编码这部分思想应该来源于Transformer中的位置编码，图中坐下虚线框描述了这部分，分别记录了X Y 坐标跟 ISO 信息，经过一个3x3 卷积提取特征后直接跟图像特征相加
	2. Unet 中的标准卷积快被替换为 NAFBlock 引入通道注意力。[[../../machine learning/image restoration/2022 NAFNet Megvii|2022 NAFNet Megvii]]
			_“We adopt UNet as the base architecture, **introduce the Nonlinear Activation Free block** in [4] to replace ordinary convolutional blocks”_
	3. 跳跃连接与瓶颈层中的转置自注意力,通过计算通道维度的协方差矩阵来捕获所有像素间的全局依赖关系，解决暗角噪声（Dark-shading）的长程空间相关问题
			_“First, we introduce interaction blocks into the shortcut connections and latent layers of the U-Net-based DenoiseNet to enable global information exchange. **These blocks employ the transposed self-attention mechanism**, which enables the network to explicitly capture correlations between all pixel pairs, effectively overcoming the receptive field limitation and achieving genuine global information fusion.”_
2. **ColorNet** 预测 **AWB增益** 跟3x3的CCM矩阵：他是一个轻量网络，直接在raw上提取信息，统计颜色的分布信息。（不去理解图像内容）

# 网络训练

网络训练的核心就是**解耦**：两个网络的难度不同，收敛速度有差异，一起训练会不稳定。
> *“To prevent inadequate training due to asynchronous progress”*
## step1 训练 DenoiseNet

颜色部分依赖降噪网络的输出，因此先训练降噪网络也是符合常理：
- **输入**：带噪的 RAW 图像。
- **GT**：长曝光或合成的干净 RAW（Ground Truth）。
- **损失函数**：主力 RAW 域 L1 损失，辅助 RGB 域 L1 损失（RGB 使用真实 WB、CCM 参数）
- **关于反向传播**：ISP 中的CCM WB gamma 都是可导的，demoniac 大部分算法都是不可导的，但是本文用的[[../ISP/demosaic/2004 Malvar-He-Cutler Microsoft|2004 Malvar-He-Cutler Microsoft]]是可导的demoniac 方法（也考虑到了梯度，但效果相对非线性算法差点，不过本文主要针对暗光场景，主要矛盾不在demoniac）
## step2 训练 ColorNet

- **状态**：**DenoiseNet 被完全冻结（Frozen）**，确保了 ColorNet 的输入特征是稳定的。
- **输入**：训练好的 DenoiseNet 输出的“干净 RAW”。
- **输出**：12 维向量（3 维 WB + 9 维 CCM）。
- **损失函数**：
	1. **直接监督**：预测参数与 GT 参数的 L1 距离。
	2. **间接监督**：RGB 域 L1 损失
