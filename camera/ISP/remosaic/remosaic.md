---
type: note
status: draft
tags:
  - camera
  - remosaic
rating: 0
create: 2026-06-29
update: 2026-08-16
---
remosaic 技术相关的资料比较少，通常是sensor内部完成，因此是各个厂家核心技术。对外纰漏的细节不多。

目前remasoic 主要有两种实现方式：
- 传统算法，思路与demosaic 算法类似，都是利用色差信号缓慢变化的特点+梯度识别边缘纹理，进行插值
- NN 算法，目前的主流方案。sony 的公开文档显示 已经将AI remosaic 做到sensosensor 内部了

# 传统算法
硬件remosaic [Quad Bayer Coding | Image Sensor for Mobile | Technology | Sony Semiconductor Solutions Group](https://www.sony-semicon.com/en/technology/mobile/quad-bayer-coding.html)

方向梯度插值（2012专利）：[U.S. Patent for Image processing device, and image processing method, and program Patent (Patent # 9,179,113 issued November 3, 2015) - Justia Patents Search](https://patents.justia.com/patent/9179113)

# AI remosaic

## sony sensor 内的 AI remosaic
[索尼发布约2亿像素AI移动图像传感器LYTIA 901，单摄可实现高清变焦](https://www.sony.com.cn/content/sonyportal/zh-cn/cms/newscenter/techonology/2025/20251127.html)
![[attachments/Pasted image 20260629120033.png]]

## NN remosaic 
MIPI 2022 remosaic 挑战赛 [[2209.07060] MIPI 2022 Challenge on Quad-Bayer Re-mosaic: Dataset and Report](https://arxiv.org/abs/2209.07060)
这里关注下训练数据怎么得到的，下图是生成数据的方式：
- 虽然原始数据是 QBC 类型的，但是生成数据链路上做了 binning，相当于跟普通的bayer raw 一致的。
- 拿到bayer raw 之后用 demosaic_net 重建为RGB 图像，在这个RGB 图像上进行不同的采样，分别得到:
	- QBC 类型的raw，输入数据
	- bayer 类型的 raw，真值
![[attachments/Pasted image 20260629165005.png]]
