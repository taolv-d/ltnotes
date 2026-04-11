论文：[Rethinking Reconstruction and Denoising in the Dark: New Perspective, General Architecture and Beyond](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Rethinking_Reconstruction_and_Denoising_in_the_Dark_New_Perspective_General_CVPR_2025_paper.pdf)

# 本文指出当前 nn isp 存在的问题：
![[attachments/Pasted image 20260409152230.png]]
1. 早期只能二选一，要么raw->raw 要么raw到rgb
2. 中间版本是串联，但浪费计算量（两个网络有共享的部分）、误差积累、前后两个网络目标不一致
3. 本文方案：用一个强backbone提特征，接两个head 出raw 跟rgb

# 本文的注意力机制：
![[attachments/Pasted image 20260409152309.png]]

对于色彩：利用池化来缩小图像（只关注低频的颜色），同时添加了可学习的参数，因此引入了低频注意力，然后再上采样回去。
对于细节：split生成两张图，一个原图、一个可学习的mask，mask 保留想要的细节，移除噪声，相当于一个注意力机制。

# 双头训练如何避免打架：
作者意识到训练冲突的问题，他没有设计一个精巧的传统算法去避免冲突的问题。而是设计了一个交互网络，让网络自己去学习如何处理冲突问题
![[attachments/Pasted image 20260409153155.png]]