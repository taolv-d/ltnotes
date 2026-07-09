---
type: artical
status: done
tags:
  - camera
  - nn-isp
rating: 0
create: 2026-04-14
update: 2026-07-03
publish: 2025
url: https://arxiv.org/abs/2512.08564
---
# Modular Neural Image Signal Processing

github: [GitHub - mahmoudnafifi/modular_neural_isp: Code for the paper: "Modular Neural Image Signal Processing". A modular neural ISP with interpretable stages, multi-style rendering, cross-camera generalization, and post-editable re-rendering with an interactive photo editor. · GitHub](https://github.com/mahmoudnafifi/modular_neural_isp)

![[attachments/Pasted image 20260703200147.png]]

三星的这个工作把传统 ISP 拆成几段“可学习但可控”的模块：

>  raw 预处理（BLC demosaic） 
> raw denoising 
> AWB/CCM color correction 
> linear sRGB 
> photofinishing 
> guided upsampling 
> enhancement 
> sharpening 
> JPEG保存/可选raw回嵌

其中可学习部分主要是：raw 降噪、风格渲染、增强，我们主要聚焦这些部分的实现方式以及训练方法，尤其是风格化部分。
JPEG/raw 回嵌这里与 ISP 主题差异比较大，这里不展开。

# 可学习模块

**raw denoising** : [[../../mach[[../../machine learning/image restoration/2022 NAFNet Megvii|NAFNet]] 架构全卷积网络，通常跟着相机走，论文搞了一个泛化性好一些的来适应
**photofinishing 风格渲染**，顺序预测并应用一组 ISP 风格参数（下采样省算力）：
1. GainNet：预测数字增益
2. GlobalToneMappingNet：预测全局 tone mapping 参数
3. LocalToneMappingNet：预测局部 tone mapping 系数
4. LUTNet：预测二维 CbCr chroma LUT
5. GammaNet：预测 gamma
**enhancement**  NAFNet 做细节增强。

## denoise & enhance
两个NAFNet 都是独立监督训练，降噪可以通过教师模型来生成伪GT数据，增强部分类似。另外为了速度，降噪的参数量大一些，增效小很多，与任务难度有关（降噪本身难度大，还要泛化）。

## photofinishing

photofinishing 部分更有意思，他们预测ISP参数，应用到传统ISP算法上。同时训练也是端到端，值得深入看看。

| 网络名称           | 参数量   | 架构特点                                               | 用途                   |
| -------------- | ----- | -------------------------------------------------- | -------------------- |
| **Gain Net**   | ~6.5K | **极轻量**：CNN + MBConv + CA → 全局平均池化 → 全连接 → Sigmoid | 预测**1个全局标量**（增益因子）   |
| **Gamma Net**  | ~6.5K | **与Gain Net完全相同**的架构                               | 预测**1个全局标量**（伽马值）    |
| **GTM Net**    | ~28K  | 类似但更宽：CNN + MBConv + CA → 多级池化 → 全连接 → Softplus    | 预测**3个全局参数** (a,b,c) |
| **LTM Net**    | ~120K | **最复杂**：双分支（多尺度引导子网络 + 网格预测子网络）                    | 预测**空间变化的系数图**（5张图）  |
| **Chroma Net** | ~45K  | 独特设计：可微分直方图 → 编码器-解码器（含亮度引导注意力）                    | 预测**2D色度LUT**        |
### LTM Net

该模块设计要求：1. 能处理局部差异 2. 高效 3. 可解释
1. 为了高效，网络输出一个64 * 64 的系数网格，插值后应用到全图
2. 接收 **两张图**，即GTM前后的图，这样就知道怎么修补
3. loss 要求亮度接近GT、平滑、无Halo

下图是网络架构：
- 上半部分是多尺度引导子网络，分别提取原分辨率、1/2分辨率、1/4分辨率的**亮度信息**，融合后输入给后级网络。目的就是让后级网络看到不同尺度的亮度信息
- 下半部分会拼接GTM后的图作为输入，多次下采样到64 * 64大小
![[attachments/Pasted image 20260703202941.png]]

### Chroma Net

这部分网络结构更加复杂：
1. **Differentiable Histogram**（可微分直方图）：将 CbCr （$C_{LTM}$）色度通道的分布压缩成一个紧凑的直方图表示，作为网络的输入
2. Hist 从直方图提取色度特征
3. Encoder
4. Luminance-Guidance(亮度引导子网络)，接收亮度信息（$Y_{LTM}$）,目的是根据亮度信息影响色调映射行为，输出注意力向量
5. Luminance-Guidance 输出与encoder输出逐通道相乘，实现注意力
6. Decoder 生成**残差**LUT
7. Base LUT：**图像无关的、全局色度映射基线**，残差 LUT 只负责在这个基础上做**微调**。这种设计让训练更稳定，也降低了网络的学习负担（注意也是可学习的）
8. 最终的LUT就是 Base LUT + 残差LUT

这里的设计哲学是：
1. 只改色度，亮度用来引导（注意力），根据亮度调整色调映射行为
2. 残差学习，降低难度

![[attachments/Pasted image 20260703203734.png]]

### 端到端训练

风格化模块的训练非常有意思：他是一个端到端的多任务学习。
**训练数据**：输入经过色彩校正的线性 sRGB 图像（降采样到1/4分辨率）。监督是各种风格的图像
**损失函数**：
- **底层损失**：L1, SSIM, CbCr Loss
- **感知损失**：VGG,  ΔE (色差，CIE Lab空间)
- **正则化损失**：LUT Smoothness, LTM Smoothness, Tone Mapping Loss, Luminance Consistency

**反向传播**，为了反向传播，这一块必须要**可微分**：
1.  Gamma：天然可微分，他的数学公式是$I'=I^{1/\gamma}$
2. LUT: 离散映射，**直接查表是不可微的**，但Chroma Net和3D RGB LUT都通过**插值（Interpolation）** 解决了这个问题，从而实现了梯度回传。
> **原文依据 (Sec. C.6)**：  
> “The LuT is learned in an end-to-end fashion and applied via **differentiable grid sampling** to enable backpropagation through the photofinishing module.”

3. 其他 gain、GTM 的 S 曲线等均可以直接求导数，或者利用差值实现