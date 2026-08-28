---
type: artical
status: done
tags:
  - backbone
  - attention
  - conv
rating: 0
create: 2026-04-21
update: 2026-08-28
publish: 2022-01-01
url: https://arxiv.org/pdf/2201.03545
---
原文：A ConvNet for the 2020s
**ConvNeXt** 是一种现代 CNN 网络架构（backbone）。他想在ResNet中引入现代化的设计（但不引入transformer 的注意力机制），使纯卷积的模型达到ViT^[[ViT]], Swin^[[Swin]] 的水平。
# 模型结构探索
本文作者没有提出一个天才般的设计，而是对标准 ResNet 改进，向 Transformer 类网络效果靠近。
主要改进策略有：
1. 现代化训练策略：AdamW、Mixup、Cutmix、RandAugment、随机擦除、随机深度、标签平滑
2. 宏观设计：
	1. 调整阶段计算比例：将ResNet-50的block分布从(3,4,6,3)改为(3,3,9,3)，匹配Swin-T的比例。
	2. 修改Stem：将ResNet的7×7卷积+最大池化，替换为ViT风格的“Patchify”（4×4、stride 4卷积）。
3. 引入分组卷积（ResNeXt-ify）：将3×3卷积替换为**深度可分离卷积**，并增加网络宽度（64→96，匹配Swin-T）,这里分组卷积有点类似多头注意力机制，可能某一组学到了注意力![[attachments/Pasted image 20260828224925.png]]
4. 倒置瓶颈：将ResNet的瓶颈结构（大通道→小通道→大通道）反转，变为**小通道→大通道→小通道**（类似MobileNetV2和Transformer MLP）。下图中a 是经典的“大小大”，b是“小大小”，c是DSC 上移后的最终结果![[attachments/Pasted image 20260828225235.png|523]]
5. 大卷积核
	1. 先移动depthwise conv位置：将其从瓶颈中间移到前面（类似Transformer中MSA在MLP之前）。好处是：1. 增大卷积核会增大计算量，不过主要集中在第一层，改为DSC可以降低计算量。2. 更贴近transform的MLP。
	2. 增大卷积核：发现**7×7** 效果最佳。
6. 微观设计
	- **激活函数**：ReLU → GELU（精度不变）。
	- **减少激活函数数量**：每个block只保留一个GELU（像Transformer）。
	- **减少归一化层**：每个block只保留一个BN（在1×1卷积前）。
	- **BN → LN**：批量归一化替换为层归一化。 
	- **独立下采样层**：在stage之间加入独立的2×2、stride 2卷积，并增加LN辅助稳定训练。
![[attachments/Pasted image 20260828224612.png|445]]