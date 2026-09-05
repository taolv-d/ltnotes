---
type: note
status: done
tags:
  - isp
  - calibration
rating: 0
create: 2026-09-04
update:
---
ISP 标定主要包括以下这些方面：
# LSC
lsc 标定可以说是ISP中最重要的标定模块了（主要是广角，长焦影响小一点）。参考：[[LSC]]
# AWB
1. AWB 标定主要是补偿不同模组间的 R/G B/G 差异，**将所有模组对齐到golden**。标定通常也在D50光源下标定。
2. 还有另外一个AWB 白点标定，通常是对golden模组进行的，参考：[[AWB/AWB|AWB]]
# QSC
QSC 主要针对 Quad bayer sensor 标定（[[../sensor/2019 QBC 2x2OCL sony|2019 QBC 2x2OCL sony]]）。主要是补偿四个小像素的灵敏度差异（sensor 自身有轻微差异，同时CRA，尤其广角，会在边缘引入灵敏度差异）
![[attachments/Pasted image 20260904232806.png|527]]
# SPC
SPC 主要真对 PD 像素，左右pixel 如果有感度差异，也会影响PD对焦的性能。具体见：[[../AF/PDAF|PDAF]] 关于标定的介绍（通常sensor 厂会提供标定方法、环境、算法库）
# BLC
BLC 主要是sensor的暗电平。通常需要每个gain都标定。现代sensor BLC 通常都比较稳定（sensor 内部有暗电平的矫正），通常可以通过寄存器设置想要的BLC（BLC 不直接搞为0 还是为了噪声的完整性，不能把负数部分的噪声全都阶段。BLC 设太大会影响动态范围，一般8bit下BLC=8）

blc 不矫正其实本身问题不大。但是ISP中很多 gain 操作就会有影响，gain 操作期望最小有效信号为0，这样所有信号都是等比例放大。不是0就会出问题（偏色 / 发灰等）
# BPC
坏点标定，记录坏点坐标，一般需要暗场/亮场都标定，记录亮暗坏点。
# CA
CA 主要标定色差，属于比较困难的标定，而且往往标定效果不理想。很多 ISP 供应商都建议用golden值。具体介绍见[[CA]]

# AF 相关标定
出了前面介绍的PD像素的标定。AF 还有：
1. 马达位置的标定：标定马达位置与对焦距离
2. DCC 标定、gain map 标定，参考：[[../AF/PDAF|PDAF]] 
高通PDAF标定手册：https://usermanual.wiki/Pdf/80NV1251PDAFModuleCalibrationGuide.750511322.pdf