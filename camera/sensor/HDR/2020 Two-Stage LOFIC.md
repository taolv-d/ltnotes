---
type: artical
status: done
tags:
  - 
rating: 0
create: 2026-07-12
update: 2026-07-13
publish: 2020-01-01
url: https://xplorestaging.ieee.org/document/9281334
---
本文源自论文：An Over 120 dB Single Exposure Wide Dynamic Range CMOS Image Sensor With Two-Stage Lateral Overflow Integration Capacitor

[这篇博客有论文更多图片/视频介绍](https://zhuanlan.zhihu.com/p/2015919843690430701)

这篇文章的对 LOFIC 改进非常简单，他把原来一个LOFIC电容分成两个，因此有三个转换增益。
他的原理图如下：
**电荷的溢出路径如下：PD->FD->LOFIC1->LOFIC2**
![[attachments/Pasted image 20260712235817.png|465]]

![[attachments/Pasted image 20260713000046.png|503]]
他的工作时序如上图所示（会读FD， FD+LOFIC1， FD+LOFIC1+LOFIC2）：
- t8->t1: R T S1 S2 打开，复位
- t1: R T 关闭读取复位噪声，包括FD，LOFIC1，LOFIC2，记为**N3**
- t2: R T S2 关闭，读取复位噪声，包括FD，LOFIC1，记为**N2**
- t3: R T S1 S2 均关闭，曝光
- t4： R T S1 S2 均关闭，读取复位噪声，仅包含FD，记为**N1**（这里没有溢出，读取的是噪声，如果溢出了，则是溢出的信号值）
- t5: T 打开，PD 中电荷转移到FD
- t6: 读取FD 内信号值 **S1+N1**，即**HCG**转换结果
- t6-t7: T S1 都打开，FD LOFIC1 两个电容内电荷混合
- t7: S1 打开，读取 FD+LOFIC 信号：**S2+N2**，即**MCG**转换结果
- t7-t8: T S1 S2 打开，PD FD LOFIC1 LOFIC2 电荷混合
- t8: 读取 FD，LOFIC1，LOFIC2 信号：**S3+N3**,即**LCG** 转换结果