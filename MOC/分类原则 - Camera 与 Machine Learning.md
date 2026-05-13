# 分类原则 - Camera 与 Machine Learning

核心规则：问题域优先，方法其次。

## 放到 camera

以相机成像链路为中心的内容放 `camera`。只要核心问题离不开镜头、sensor、RAW、ISP、IQ、标定、低光、HDR、降噪、Neural ISP，即使用了神经网络，也放这里。

例子：PMRID、YOND、VST、Retinex、Low-Light Vision、Neural ISP、RAW 多帧复原、ISP tuning。

## 放到 machine learning

脱离相机也成立的通用方法放 `machine learning`。

例子：U-Net、ResNet、ViT、MobileNet、DSC、SE、Coordinate Attention、LPIPS、M-估计量、Diffusion 基础、GAN 基础。

## 交叉内容

一篇笔记只放一个物理位置，交叉关系用链接或 MOC 表达，不复制内容。

判断口令：
- 问题是相机链路里的问题吗？是 -> `camera`
- 文章重点是通用模型机制吗？是 -> `machine learning`
- 两边都有？看标题和正文主要回答哪个问题。