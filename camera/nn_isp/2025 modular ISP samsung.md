---
type: artical
status: draft
tags:
  - camera
  - nn-isp
rating: 0
create: 2026-04-14
update: 2026-07-03
publish: 2025
url: https://arxiv.org/abs/2512.08564
---

文章：Modular Neural Image Signal Processing
github: [GitHub - mahmoudnafifi/modular_neural_isp: Code for the paper: "Modular Neural Image Signal Processing". A modular neural ISP with interpretable stages, multi-style rendering, cross-camera generalization, and post-editable re-rendering with an interactive photo editor. · GitHub](https://github.com/mahmoudnafifi/modular_neural_isp)

![[attachments/Pasted image 20260703200147.png]]

这个仓库把传统 ISP 拆成几段“可学习但可控”的模块

```
-> raw 预处理（BLC demosaic） 
-> raw denoising 
-> AWB/CCM color correction 
-> linear sRGB 
-> photofinishing 
-> guided upsampling 
-> enhancement 
-> sharpening 
-> JPEG保存/可选raw回嵌
```

可学习部分：

**raw denoising** : [[../../mach[[../../machine learning/image restoration/2022 NAFNet Megvii|NAFNet]] 架构全卷积网络，通常跟着相机走，论文搞了一个泛化性好一些的来适应

**photofinishing 风格渲染**，顺序预测并应用一组 ISP 风格参数（下采样省算力）：
1. GainNet：预测数字增益
2. GlobalToneMappingNet：预测全局 tone mapping 参数
3. LocalToneMappingNet：预测局部 tone mapping 系数
4. LuTNet：预测二维 CbCr chroma LUT
5. GammaNet：预测 gamma

**enhancement**  NAFNet 做细节增强。

模型的架构与训练

两个NAFNet 都是独立监督训练，降噪可以通过教师模型来生成伪GT数据。增强部分类似

photofinishing 部分更有意思，他们预测ISP参数，先看看网路怎么实现的：

| 网络名称           | 参数量   | 架构特点                                               | 用途                   |
| -------------- | ----- | -------------------------------------------------- | -------------------- |
| **Gain Net**   | ~6.5K | **极轻量**：CNN + MBConv + CA → 全局平均池化 → 全连接 → Sigmoid | 预测**1个全局标量**（增益因子）   |
| **Gamma Net**  | ~6.5K | **与Gain Net完全相同**的架构                               | 预测**1个全局标量**（伽马值）    |
| **GTM Net**    | ~28K  | 类似但更宽：CNN + MBConv + CA → 多级池化 → 全连接 → Softplus    | 预测**3个全局参数** (a,b,c) |
| **LTM Net**    | ~120K | **最复杂**：双分支（多尺度引导子网络 + 网格预测子网络）                    | 预测**空间变化的系数图**（5张图）  |
| **Chroma Net** | ~45K  | 独特设计：可微分直方图 → 编码器-解码器（含亮度引导注意力）                    | 预测**2D色度LUT**        |

**LTM Net**
要满足：1. 能处理局部差异 2. 高效 3. 可解释
1. 为了高效，网络输出一个64 * 64 的系数网格，插值后应用到全图
2. 接收 **两张图**，即GTM前后的图，这样就知道怎么修补
3. loss 要求亮度接近GT、平滑、无Halo

下图是网络架构：
- 上半部分是多尺度引导子网络，分别提取原分辨率、1/2分辨率、1/4分辨率的**亮度信息**，融合后输入给后级网络。目的就是让后级网络看到不同尺度的亮度信息
- 下半部分会拼接GTM后的图作为输入，多次下采样到64 * 64大小

![[attachments/Pasted image 20260703202941.png]]

**Chroma Net**
这部分更加复杂：
1. **Differentiable Histogram**（可微分直方图）：将 CbCr （Cltm）色度通道的分布压缩成一个紧凑的直方图表示，作为网络的输入
2. Hist 从直方图提取色度特征
3. Encoder
4. Luminance-Guidance(亮度引导子网络)，接收亮度信息（Yltm）,目的是根据亮度信息影响色调映射行为，输出注意力向量
5. Luminance-Guidance 输出与encoder输出逐通道相乘，实现注意力
6. Decoder 生成**残差**LUT
7. Base LUT：**图像无关的、全局色度映射基线**，残差 LUT 只负责在这个基础上做**微调**。这种设计让训练更稳定，也降低了网络的学习负担（注意也是可学习的）
8. 最终的LUT就是 Base LUT + 残差LUT

这里的设计哲学是：
1. 只改色度，亮度用来引导（注意力），根据亮度调整色调映射行为
2. 残差学习，降低难度

![[attachments/Pasted image 20260703203734.png]]


4. **Photofinishing 模块（Gain/GTM/LTM/Chroma/Gamma）**：这是**最核心的训练环节**，采用的是**端到端的多任务学习**。
    
    - **输入**：已经过色彩校正的线性 sRGB 图像（降采样到1/4分辨率）。
        
    - **监督信号**：对应风格（Style #0 ~ #5）的目标 sRGB 图像。
        
    - **损失函数**：论文精心设计了一个**组合损失函数**（Eq. 9）：
        
        - **底层损失**：`L1`, `SSIM`, `CbCr Loss`
            
        - **感知损失**：`VGG Perceptual Loss`, `ΔE Color Loss` (CIE Lab空间)
            
        - **正则化损失**：`LUT Smoothness`, `LTM Smoothness`, `Tone Mapping Loss`, `Luminance Consistency`
            
    - **关键点**：虽然“Photofinishing模块”本身是端到端训练的，但由于**损失函数的设计**（如TM Loss、Luma Loss、LTM Smoothness）对每个子模块施加了特定的约束（如GTM负责全局亮度，LTM负责局部细节），使得最终每个子模块能各司其职，而不是互相“抢活”。


**整个端到端训练过程中，反向传播的梯度可以顺畅地流过所有模块，包括Gamma和LUT（查找表）操作。**

为什么能做到这一点？关键在于它们都被设计成了**完全可微分（Differentiable）** 的操作。下面我为你逐一拆解：

---

### 1️⃣ Gamma校正：天然的“可微分”操作

Gamma校正本身就是一个简单的数学函数：

Igamma=Ichroma(1/γ)Igamma​=Ichroma(1/γ)​

- **正向传播**：对每个像素的RGB值进行幂运算。
    
- **反向传播**：这是一个初等函数，其导数 ddxxn=n⋅xn−1dxd​xn=n⋅xn−1 是明确定义且可直接计算的。
    
- **结论**：只要 `gamma` 参数是可学习的标量（由 `D_gamma` 网络预测），整个操作就是完全可微的。**前向传播进行计算，反向传播梯度直接穿透幂函数**，更新 `D_gamma` 网络的权重。
    

---

### 2️⃣ LUT（查找表）：通过“插值”实现可微分

查找表（LUT）本身是一个离散映射，**直接查表是不可微的**，但Chroma Net和3D RGB LUT都通过**插值（Interpolation）** 解决了这个问题，从而实现了梯度回传。

#### Chroma Net 的 2D 色度 LUT（CbCr空间）

- **正向传播**：
    
    - 输入像素有一个 `(cb, cr)` 坐标。
        
    - Chroma Net 输出一个 24×24×2 的 LUT。
        
    - 在 LUT 的 24×24 网格中，`(cb, cr)` 坐标会**落在四个相邻的格子点之间**。
        
    - 输出值 = 对这四个格子点的值进行**双线性插值（Bilinear Interpolation）**。
        
- **反向传播**：
    
    - **梯度可以回传**：因为双线性插值是一个连续的线性操作，其导数是**插值权重本身**。
        
    - **结论**：输出值对四个相邻格子点的导数 = 对应的插值权重。因此，**梯度可以从最终损失，通过双线性插值，直接流回LUT的四个格子点**，再进一步流回Chroma Net的权重。
        

> **原文依据 (Sec. C.6)**：  
> “The LuT is learned in an end-to-end fashion and applied via **differentiable grid sampling** to enable backpropagation through the photofinishing module.”

#### 3D RGB LUT（可选，用于艺术风格）

- **原理完全相同**：它在一个 11×11×11 的 3D 网格上进行**三线性插值（Trilinear Interpolation）**。
    
- **反向传播**：梯度通过三线性插值的权重，流回相邻的 8 个格子点，再流回网络。
    

---

### 3️⃣ 整个 Photofinishing 模块的梯度流

在端到端训练中，整个 Photofinishing 模块（包含 Gain、GTM、LTM、Chroma、Gamma）是作为一个整体联合优化的：

text

输入 (I_LsRGB)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Photofinishing 模块                          │
│                                                                     │
│  1. Digital Gain (D_gain)  ──► 可微分乘法                           │
│         │                                                           │
│         ▼                                                           │
│  2. GTM (D_GTM) ──► 可微分S型曲线函数                               │
│         │                                                           │
│         ▼                                                           │
│  3. LTM (D_LTM) ──► 可微分S型曲线 + 空间插值                       │
│         │                                                           │
│         ▼                                                           │
│  4. Chroma Net ──► 可微分直方图 → CNN → 2D LUT → 双线性插值        │
│         │                                                           │
│         ▼                                                           │
│  5. Gamma (D_gamma) ──► 可微分幂函数                               │
│         │                                                           │
│         ▼                                                           │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
输出 (I_gamma) ──► 计算 Loss ──► 反向传播梯度

**关键点**：当 Loss 计算完并开始反向传播时，梯度会沿着这条链**逐级回传**：

- 从 `I_gamma` → 反向穿过 **Gamma** 的幂函数 → 更新 `D_gamma`
    
- 继续穿过 **Chroma Net** 的 LUT → 更新 `D_chroma`
    
- 继续穿过 **LTM** 的 S型曲线和系数网格 → 更新 `D_LTM`
    
- 继续穿过 **GTM** 的 S型曲线 → 更新 `D_GTM`
    
- 继续穿过 **Digital Gain** → 更新 `D_gain`
    

整个链条中的所有操作都是可微的，因此**端到端训练完全可行**。