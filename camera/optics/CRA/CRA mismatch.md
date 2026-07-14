---
type: note
status: done
tags:
  - camera
  - optics
  - cra
rating: 0
create: 2026-04-09
update:
---

CRA 的前置知识参考 [[../../sensor/CMOS image sensor（CIS）|CMOS image sensor（CIS）]]  以及 [[CRA/CRA]]

图片来自：[射线光学模拟 - PhyDemo](。https://phydemo.app/ray-optics/cn/)

## Luma shading、color shading

![[../attachments/Pasted image 20260529182849.png]]

上图是一个通过shift micro lens 改变sensor CRA 的示意图（这里位于整个sensor的左侧，画面底部的棕色部分为sensor的感光位置）。图中三束光线分别为：
- 中间光束是CRA match 状态的，他的光线打到自己的感光区域上。
- 左边光束 是 **lens CRA 偏小** 的情况，会出现 **Luma Shading**。此时光线更直，经过micro lens后会偏离自己的感光区域，打到不感光的区域上，对应图像亮度下降。（lens的CRA不可能为负，因此不会找到右侧的感光区域上）
- 右边光束是 **lens CRA 偏大**的情况，会出现 **Color Shading**。micro lens 的折射能力无法把这么大角度的光聚到自己的感光区域上。甚至会斜着打到左侧的感光区域引起crosstalk，对于彩色sensor 就会出现画面边缘颜色不对的问题

color shading 的真实表现可以参考下图：
![[attachments/2024-01-07-16-12-42-image.png]]
![[attachments/2024-01-07-16-11-53-image.png]]

## CRA mismatch 常见的几类问题

**故障现象：边缘紫边/绿边**  
  
· 核心原因：镜头CRA大于芯片CRA，光线串扰。  
· 解决思路：换CRA更小的镜头，或启用最强算力的去紫边算法。  
· 修复难度：极难  
  
**故障现象：四周渐晕暗角**  
  
· 核心原因：镜头CRA过小，光通量损失。  
· 解决思路：ISP开启LSC（镜头阴影校正）补偿。  
· 修复难度：容易  
  
**故障现象：中心清晰，边缘模糊**  
  
· 核心原因：CRA不匹配导致光线无法有效收集，系统边缘解像力大幅下降。需注意与镜头自身光学素质不佳做区分——前者镜头本身MTF可能正常，问题出在CRA匹配上。  
· 解决思路：优先核对镜头与芯片CRA曲线是否匹配；如匹配正常，再排查镜头本身解像力。  
· 修复难度：困难  
  
**故障现象：高光区域像“染色”了一样**  
  
· 现象描述：典型表现有两种——明暗交界处的高光边缘挂上紫色或绿色“镶边”，高亮物体的颜色向外溢染；大面积过曝区域内部颜色不均匀，如白墙过曝处中心偏红、边缘偏绿。  
· 核心原因：高光场景下，镜头CRA大于芯片CRA，超出微透镜有效收光角范围。强光斜射入相邻像素，造成像素间颜色串扰。  
· 解决思路：换CRA更小的镜头；或与Sensor厂联合调试微透镜偏移方案；ISP端启用去紫边/高光溢出抑制算法。  
· 修复难度：困难

此外，CRA mismatch 也会影响到PD，具体见 [[../../AF/PDAF|PDAF]]
