论文：[Rethinking Reconstruction and Denoising in the Dark: New Perspective, General Architecture and Beyond](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Rethinking_Reconstruction_and_Denoising_in_the_Dark_New_Perspective_General_CVPR_2025_paper.pdf)

Unet + attaction
# 本文指出当前 nn isp 存在的问题：
![[attachments/Pasted image 20260409152230.png]]
1. 早期只能二选一，要么raw->raw 要么raw到rgb
2. 中间版本是串联，但浪费计算量（两个网络有共享的部分）、误差积累、前后两个网络目标不一致
3. 本文方案：用一个强backbone提特征，接两个head 出raw 跟rgb

# 本文的注意力机制：
![[attachments/Pasted image 20260409152309.png]]

上图是 CNPModule 的框图（色彩和噪声感知）
U-Net 是整体骨架，CNPModule 是骨架里每一层反复用的小处理单元

**对于色彩**（GCP: Global Chromatic Perceptor）：利用池化来缩小图像（只关注低频的颜色），同时添加了可学习的参数，因此引入了低频注意力，然后再上采样回去。

**对于细节** （RDE: Refined Detail Extractor）：split生成两张图，一个原图、一个可学习的mask，mask 保留想要的细节，移除噪声，相当于一个注意力机制。

### GCP
 
它先把特征变成三份 c1,c2,c3，然后把 c1,c2 池化到 2x2。

它不是标准 Transformer 那种“空间位置和空间位置做注意力”，而更像“通道和通道做关系建模”。

```
attn = (c1 @ c2.transpose(-2,-1))  
```

前面 rearrange 之后张量形状是 B x C x H x W，所以乘出来是 B x C x C。  
也就是说，它学的是“哪些通道应该互相强调”，而不是“左上角像素看右下角像素”。

为什么这样设计合理？  
因为 RAW 去噪/重建里，颜色通道关系很重要。  
R、G、B 之间的统计联系、噪声分布差异、颜色一致性，比纯空间远距离关系更关键。

### RDE

轻量细节增强块：局部纹理要保留，但噪声不能被一起放大，所以加一个门来筛。

1. 先做 7x7 depthwise conv，提局部空间信息。
2. 再做 LayerNorm。
3. 再过 SpatialGatingUnit，把特征分两半后相乘，相当于做门控筛选。
4. 最后加残差。

# 模型细节

### 两个 Head 训练如何避免打架

两个头轻量交互，自己学习参数避免打架
RGB_Head 和 RAW_Head 都不是直接拿 backbone 特征出图，而是先做一次轻量交互：

1. 每个 head 从共享特征 x 里各自生成 q
2. 主模型从共享特征 fea 里生成共享的 k,v
3. compute_attn(x,q,k,v) 让 head 用自己的 query 去“读取”共享 value

这就是论文里说的双头交互/任务关联的落地点之一。

而且这里也不是标准 token attention。  
q,k 被池化到 2x2，所以 attn 仍然主要是 C x C 的通道关系矩阵。  
但 v 没有池化，还是全分辨率，所以它相当于：  
“用一个很小的全局通道关系表，去调制整张特征图。”

这招挺巧：

1. 计算量小
2. 能把全局颜色/噪声先验注入到 head
3. RGB 和 RAW 两个头共享 k,v，说明它们看的“公共知识”是同一份，但各自 query 不同，所以关注重点不同

### 一张图在 CANS_Plus 里怎么流动

1. x = _check_and_padding(x)
2. fea = backbone(x)
3. k, v = get_kv(fea).chunk(2)
4. k = adaptive(k) 变成 2x2
5. raw = raw_head(fea, k, v)
6. rgb = rgb_head(fea, k, v)
7. crop 回原图大小

```
PackRaw -> (B, 4, 512, 512)
	黑电平校正 + 归一化到 [0,1]。
	按 Bayer 位置把一个像素网格拆成 4 个子图，组成 [R, G1, B, G2] 四通道。
conv ->  (B, 32, 512, 512)
encoder1 -> (B, 64, 256, 256), enc1 (B, 32, 512, 512)
encoder2 -> (B, 128, 128, 128), enc2 (B, 64, 256, 256)
encoder3 -> (B, 256, 64, 64), enc3 (B, 128, 128, 128)
encoder4 -> (B, 512, 32, 32), enc4 (B, 256, 64, 64)
middle blocks 两层 -> (B, 512, 32, 32) 不改变尺寸
decoder1 -> (B, 256, 64, 64) 
decoder2 -> (B, 128, 128, 128) 
decoder3 -> (B, 64, 256, 256) 
decoder4 -> (B, 32, 512, 512) 
conv -> (B, 32, 512, 512) 即fae
get_kv(fea) -> (B, 64, 512, 512)  
	k: (B, 32, 512, 512)->(池化)adaptive(k) -> (B, 32, 2, 2) 全局摘要(2x2虽小，但有32通道)
	v: (B, 32, 512, 512)  完整细节

RAW head:
	q = get_q(fea) -> (B, 32, 2, 2)， 
	q reshape -> (B, 32, 4)
	k reshape -> (B, 32, 4)
	attn = q @ k^T -> (B, 32, 32) 即q乘k的转置
	v reshape -> (B, 32, 512 * 512)
	attn @ v 后 reshape 回 (B, 32, 512, 512) 按通道注意力，attn后的每个通道从v的各个通道拿多少信息
	conv -> raw = (B, 4, 512, 512)

rgb head:
	··· 与raw一致
	attn 后 : (B, 32, 512 * 512)
	conv1 -> (B, 32, 512, 512)
	conv2 -> (B, 12, 512, 512)
	pixel_shuffle -> rgb = (B, 3, 1024, 1024)
```


**encoder 每一层做两件事

1. 先过若干 CNPModule，尺寸不变
2. 再 Conv2d(..., kernel=2, stride=2) 存一份skip, 下采样，尺寸减半，通道翻倍

**decoder**

1. 先上采样 PixelShuffle(2)，空间翻倍，通道减半
2. 和对应 encoder 的 skip 相加， **注意这里是相加，没有concat**
3. 再过若干 CNPModule

**PixelShuffle 是怎么操作的**

PixelShuffle 常用于上采样。把“通道维”的信息重新摆到“空间维”。


 (B, C * r^2, H, W)  ->  (B, C, H * r, W * r)`

其中 r 是 upscale factor。
