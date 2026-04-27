
[[../../TODO|TODO]]

![[attachments/Pasted image 20260427173328.png]]

---

## MFSR-GAN 整体架构解读

### 🎯 任务目标

模型输入是手持智能手机拍摄的多帧低分辨率（LR）RAW 图像，输出一张高分辨率（HR）RGB 图像。整个流程分为四个大模块，下面逐一讲解：

---

### 模块一：初始去马赛克（Initial Demosaic）

图的最左侧，RAW 图像（F×H×W）首先经过一个 **Conv** 层进行初步的最近邻去马赛克（demosaic），将拜耳格式的 RAW 数据转换为初步的彩色图像。同时有一个 **Skip Connection (S)** 将原始特征绕过后续模块直接传递，用于残差融合。

---

### 模块二：多尺度基帧增强（Multi-Scale Base Frame Enhancement，左侧大蓝框）

这是架构的核心创新部分。网络将第一帧 I₁ 指定为**基帧（base frame）**，计算每个非参考帧与基帧在各尺度层级上的特征差异。

图中可以看到**三个尺度**的处理分支（分辨率依次为 F×D×H×W、F×D×H/2×W/2、F×D×H/4×W/4），每个尺度内部包含：

- **Base Frame Enhancement (BFE，橙色虚线框)**：
    - **Reference Difference Computation (RDC)**：计算各帧与基帧之间的特征差值（$RDC_1$, $RDC_2$），先在 1×D 维度计算，再扩展到 F×D 维度（Repeat, dim=0）
    - **Multi-Frame Alignment**：利用 RDC 输出的 offset features（绿色箭头），对所有帧做可变形对齐，结果通过 ⊕ 与基帧特征相加
- **MFA（Multi-Frame Attention）**：相邻两个尺度之间通过 MFA 模块做跨帧的注意力传递，将上一尺度的信息融合到下一尺度

---

### 模块三：多帧特征富集（Multi-Frame Feature Enrichment，右上蓝框）

该模块通过通道注意力机制（Channel Attention Across Frames）跨帧整合信息，结合残差连接，提升感知质量。

具体来说：

- 多尺度增强后的特征 reshape 成 1×F·D×H×W 的形状
- 送入 **Channel Attention Across Frames** 模块，跨所有帧做全局特征加权
- 结果通过 ⊕ 加上跳过的原始 reshape 特征（残差连接）
- 再 reshape 回 F×D×H×W

---

### 模块四：图像重建（Image Reconstruction，右下蓝框）

- 特征依次经过 **Conv → Reshape → Up-sample ×2**（实现 2 倍超分）
- 同时一条支路走 **Conv → Leaky ReLU** 做细化
- 两路通过 ⊕ 融合后，再经过 Skip Connection (S) 加上最初的浅层特征
- 最终送入 **Residual-in-Residual Dense Block (RRDB)**（深蓝色大块，来自 ESRGAN 的经典模块）进行精细重建

---

### 模块五：判别器（Discriminator，橙色块）

输出的 **RGB HR（3×2H×2W）** 图像同时送给 **Discriminator** 进行对抗训练，这是 GAN 的标准结构，用于提升输出图像的感知真实度（perceptual quality）。右下角可以看到一张真实的户外 HR 图像样例。

---

### 💡 核心设计思想总结

|设计|目的|
|---|---|
|强调基帧（Base Frame）|抑制多帧融合时的 ghosting（鬼影）伪影|
|多尺度 BFE|从粗到细逐步对齐，处理大位移和细节|
|RDC 差分计算|显式建模帧间运动，指导可变形对齐|
|跨帧通道注意力|自适应地从所有帧中提取有用信息|
|GAN 训练|生成更锐利、更真实的纹理细节|

在合成测试集和真实手持拍摄数据上的实验均表明，MFSR-GAN 在噪声抑制、分辨率提升和伪影消除之间取得了比现有方法更好的平衡。