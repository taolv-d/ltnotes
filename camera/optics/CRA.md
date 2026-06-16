
## CRA
CRA 的全程是 Chief ray angle，他是镜头主光线（穿过光心的线）的夹角。为什么这么关心他，原因就是主光线决定了光束聚焦的位置和角度。
![[attachments/2024-01-07-16-09-18-image.png]]
主光线 Chief ray （穿过AS中心）,边缘光线 marginal ray（穿过AS边缘）

在成像系统中讨论CRA时，我们主要关注角度，原因是 sensor 是一个三维结构，垂直射向sensor跟带角度射向sensor的光线表现是不一样的。

同理，sensor 也会针对特定角度的光线做优化，例如：
- 镜头中心的光线都是垂直射下来的，sensor 就要接收 0°光
- 边缘区域的光线是带角度射过来的，sensor就要匹配到这个角度
因此 sensor 接收光线的角度，就是sensor的CRA

| lens CRA                                       | sensor CRA                                     |
| ---------------------------------------------- | ---------------------------------------------- |
| ![[attachments/2024-01-07-14-04-59-image.png]] | ![[attachments/2024-01-07-14-05-29-image.png]] |

从上面的描述也能看到 CRA 从中心到边缘是会变换的。当然也有不变的，那就是全0°CRA，镜头光线垂直射向sensor。因此实际看 sensor 跟 lens CRA 是否匹配的时候都是整条CRA曲线是不是处处match的：
![[attachments/2024-01-07-16-16-17-image.png|325]]

CRA mismatch 会造成 luma shading 跟 color shading，因此CRA匹配需要在 3°，极限5°可能已经很难矫正了。

CRA mismatch 的影响见 [[CRA mismatch]]
此外，CRA mismatch 也会影响PD对焦 [[../AF/PDAF|PDAF]]


## CRA改进

我们当然希望 lens 出来的光线全都垂直射向 sensor，即0°CRA。这样sensor都不需要特殊处理，也不需要考虑CRA匹配的问题。**但这样的问题是 lens 的后焦很长，模组会很大**，参考下图对比

| 0°CRA镜头                                        | 手机镜头                                    |
| ---------------------------------------------- | --------------------------------------- |
| ![[attachments/2024-01-07-16-17-42-image.png]] | ![[attachments/2024-01-07-16-15-27-image.png|231]] |

因此就需要 sensor 适配镜头，从中心到边缘CRA逐渐增大。以下是几种改变sensor CRA 的手段

### sensor CRA改进手段
1. 加强微透镜折射能力
![[attachments/2024-01-07-16-18-21-image.png]]

2. pixel内隔离，避免串扰
![[attachments/2024-01-07-16-18-47-image.png]]

3. micro lens shift
![[attachments/2024-01-07-16-20-18-image.png|528]]

4. 背照式变为前照式（BSI->FSI）
![[attachments/2024-01-07-16-20-31-image.png|353]]

