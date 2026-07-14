---
type: artical
status: done
tags:
  - sensor
  - hdr
rating: 0
create: 2026-07-09
update:
publish: 2025-01-01
url: https://imagesensors.org/papers/10.60928/sm0g-j47e/
---
# A 400 × 400 3.24 μm 117 dB Dynamic Range 3-layer Stacked Digital Pixel Sensor with Triple Quantization and Fixed Pattern Noise Correction

本文来源于IISW的一片论文，介绍的是一个**DPS**(Digital Pixel Sensor（数字像素传感器）)。不过论文不会对sensor 细节过多介绍，核心的部分是下面的电路图。本文提出的sensor 是用于AR/VR 等设备的瞳孔跟踪，需要很高的动态范围。

## 三层 stack 结构

分别是：感光层、混合信号层、数字层。每个pixel 都有 ADC 电路（中间层）![[attachments/Pasted image 20260709222013.png|472]]

## 一次曝光，三次量化，HDR

1. 一次TTS，应对过曝
2. 一个FD ADC 读出，结合 logic ，应对高部
3. 一个PD ADC 对出，用于暗部

下面电路图中，左侧是一个4T 1C 的 logic 结构，额外多了一个AB 管用于防治溢出电荷污染其他pixel（即溢出电荷会经过 AB管 与电源导通 ）

右侧读出电路有三种量化工作模式，结合图中下半部分的时序图（这里原文没有过多介绍，这里先不展开深入）。有几个有趣的点可以介绍下：
1. Vin 这里实际是将 pixel 输出跟参考电压比较，TTS 模式是，参考电压固定的。ADC 时，参考电压是斜坡，对应的是斜坡型ADC
2. TTS 阶段实际也是sensor 曝光时刻，此时AB管打开，避免高光溢出

![[attachments/Pasted image 20260709220507.png|618]]

## FPN 矫正 与 ISP

该 sensor 的设计中没有CDS，同时ADC 等一致性很难保证,因此内置了FPN 矫正电路。

论文中还有FPN矫正功能，看起来像是利用OB区像素+电路实现的多帧平均来标定（动态）DSNU，以此进行矫正。

同时还能滚动矫正，以应对FPN随温度等的变化。下图左侧是 FPN 矫正前后的噪声分布变化：
![[attachments/Pasted image 20260709230642.png|602]]
另外 三种量化方式直出的信号不是平滑过渡的，会造成图像质量下降，因此sensor 内部集成ISP 来处理HDR 信号（上图右侧是ISP对信号的优化）。下图是实际拍摄的高动态场景：
![[attachments/Pasted image 20260709230950.png|598]]