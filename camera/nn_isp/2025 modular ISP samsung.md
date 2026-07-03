---
type: artical
status: done
tags:
  - camera
  - nn-isp
rating: 0
create: 2026-04-14
publish: 2025
url: 
update:
---

Modular Neural Image Signal Processing

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
**raw denoising** : [[../../mach[[../../machine learning/image restoration/2022 NAFNet Megvii|NAFNet]]M** : 未指定色温就用模型的结果

**photofinishing 风格渲染**，顺序预测并应用一组 ISP 风格参数（下采样省算力）：
1. GainNet：预测数字增益
2. GlobalToneMappingNet：预测全局 tone mapping 参数
3. LocalToneMappingNet：预测局部 tone mapping 系数
4. LuTNet：预测二维 CbCr chroma LUT
5. GammaNet：预测 gamma

**enhancement**  NAFNet 做细节增强。

