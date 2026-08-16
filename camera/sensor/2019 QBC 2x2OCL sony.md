---
type: artical
status: done
tags:
  - sensor
  - QBC
rating: 0
create: 2026-08-15
update:
publish: 2019-01-01
url: https://www.scilit.com/publications/eab5666f942f7c79008f0064f5256a8d
---
原文：A 1/2inch 48M All PDAF CMOS Image Sensor Using 0.8μm Quad Bayer Coding 2×2OCL with 1.0lux Minimum AF Illuminance Level

本文是2019 年的刚刚提出本文技术的论文，不过目前最新的技术也做了改进，可以跳转的文末看。

本文是sony 2019 年发布的传感器架构。下图是论文中与现有技术方案的对比：
![[attachments/Pasted image 20260816214229.png|598]]
其中：
- QE QBC 2x2OCL 用了更大的透镜（填充因子更高），因此量子效率更高
- PDAF 支持全像素对焦
- HDR 每个小pixel可以独立曝光（通常对角两个用相同的曝光时间）
- 分辨率（理论上能达到同尺寸bayer的分辨率，但恢复颜色有损失，因此要低于bayer 分辨率）

**2x2OCL 的crosstalk：**
1. 由于微透镜（OCL）和深槽隔离（DTI）分别由不同光刻层形成，如果发生对准偏差，光线会偏向同一颜色下的某个特定像素，导致4个同色像素的量子效率（QE）不一致，从而**降低图像分辨率**。
2. 在长波长（>500nm）下，斜射光线经过DTI分光后，可能以锐角进入相邻异色像素的DTI，导致颜色串扰。
这里 问题1 可以通过**提高制造精度**改进。问题2 主要通过**算法补偿（QSC）**。

关于 DTI，可以简单理解为隔光结构，下图左为没有DTI，右为有DTI:
![[attachments/Pasted image 20260816215342.png]]

**CRA 影响**
论文里面都是介绍sensor 自身的因素。但是sensor 做成模组后，CRA 的轻微偏差也会造成光线无法聚焦到4个小pixel中心，引入系统性的偏差。不过这也可以用QSC标定补偿。这可能也是QSC标定需要在模组状态下标定的原因。

**后续改进**
 虽然 sony 说 分辨率要比QBC效果好，但最新的结果并不是这样，包括sony也推出了改进版本，即RB2x2OCL(LYTIA610)。
 - RB像素保持2x2OCL，提供对焦能力
 - G 每个pixel 一个OCL，来提高分辨率
 ![[attachments/Pasted image 20260816221245.png|480]]

