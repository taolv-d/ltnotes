---
type: note
status: done
tags:
  - camera
  - af
  - pdaf
rating: 0
create: 2026-04-24
update: 2026-07-09
---
# PDAF 工作原理

PD的工作原理已经有很多文章讲的很清楚了，这些文章可供参考：
[片上相位检测对焦的工作原理和计算方法The Principle And Calculation Of On-Sensor Phase Detection Autofocusing - 知乎](https://zhuanlan.zhihu.com/p/658852930)

这里简短介绍PD最核心的部分：
下图是PDAF原理介绍经常会引用的一张图像，这张图像虽然介绍了PDAF的原理，但是他跟实际应用的状态有出入。真实的成像系统不会在镜头跟sensor中间加一个挡光板。
![[attachments/Pasted image 20260526221318.png|210]]
真实的sensor中PDAF的原理应该是这张图的状态（这里以全像素PD的sensor展示，其中一个微透镜下分为了左右两个PD）。这里为了简化理解，假设物体在焦面上，离焦是由于sensor到镜头之间的距离不合理导致的，实际调焦一般是动lens，很少有动sensor对焦的（动sensor 工程实现难度太大，最早的自动对焦相机实际就是动sensor，目前早已被淘汰）。此外图中省略了无穷远处入射的光线。
- 第一张图中，正确聚焦，此时来自镜头不同部位的光线被聚焦到中心pixel上，其左右pixel输出基本一致，对应到PD图上，可以看到此时左右PD图是重合的（无相位差）
- 第二张图中，sensor距离镜头过近，两个示意光线分别落在了红色pixel的右PD跟绿色pixel的左PD。此时左右PD图显然不重合（右PD图相对左PD图，图像向左侧偏移）
- 第三张图中，sensor距离镜头过远，此时也是无法聚焦，但跟图二刚好相反，对应左右PD图的偏移情况也是反过来的

![[attachments/Pasted image 20260526224001.png]]



注意，上面图中只画出了镜头边缘的两个示意性的光线，实际成像时镜头中充满观测点发出的光线，只有对上焦时才聚焦成一个点；而在两种两种离焦情况下，成像都是一个弥散斑，这也意味着离焦的时候左右PD图像不仅有相位差，同时图像还会模糊（离焦越严重，相位差越大，但图像也越糊），因此PDAF并不是万能的，当离焦很大，左右PD图太糊了，此时无法估计相位差。

此外，很多 sensor 的 PD 像素并不是全像素这种高密度的，可能只占 sensor 面积的百分之几。这也限制了PDAF的分辨率，因此PADF往往只做粗调，加上CDAF 精调完成最终的对焦

# PDAF sensor 实现

目前pdaf实现主要有三种：
1. *masked* PDAF
2. *On-Chip Lens (OCL)*
3. *Dual Photodiode (DP)*

下面介绍中的配图来源于 IISS 的CMOS 报告：[[../sensor/2021 CIS (IISS)|2021 CIS (IISS)]]

***masked* PDAF**
下图是mask PDAF 的示意图。通常它们都是左右分布的（横向纹理无法对焦），但是也有上下分布的，甚至斜向分布的，早期单反相机中的对焦传感器内部就是这种mask pixel。
这些被遮挡的pixel 通常被当作坏点，一般在ISP 的静态坏点矫正中差值（ISP 会对PD像素添加专门的模式，只需要配置PD 的起始像素、行列规律即可）。
![[attachments/Pasted image 20260706143214.png|540]]

***On-Chip Lens (OCL)***
OCL 技术是手机卷像素配套出现的技术，不过目前看来好像已经在被 DP 技术替代。
OCL 做法是让两个或者四个 pixel 公用一个微透镜，这样每个像素就是左/右PD像素了。
![[attachments/Pasted image 20260706143227.png|515]]

***Dual Photodiode (DP)***
DP 技术更近一步，把原来一个pixel 劈成左右像素，这样就能实现全像素对焦。像索尼的8 PD 技术，一个微透镜下有4个pixel（高亮场景，借助remosaic输出全分辨率，暗光时四个pixel 合并为一个pixel输出提高信噪比）。每个pixel 又分为左右PD，这样就实现了全像素对焦（不过需要标定密闭pixel 制造时的感度差异）。
关于remosaic: [[../ISP/remosaic/remosaic|remosaic]]
![[attachments/Pasted image 20260706143245.png|570]]

# PDAF 中 type 是啥

PDAF 文档中说的type1 type2 type3 主要指的是数据在哪里处理，具体参考下表：

|       | spc    | AF     |
| ----- | ------ | ------ |
| type1 | sensor | sensor |
| type2 | sensor | SoC    |
| type3 | SoC    | SoC    |

# PDAF 标定

PDAF 标定一般分为两个，一是 gain map 标定，二是 DCC 曲线标定。
## gain map

gain map 标定的目的是让左右PD图像的亮度表现一致。他类似 LSC 标定。引起PD图像不均匀的原因目前发现的有：
1. PD像素的感度差异
2. 镜头LSC引起的亮度差异

此外，**对于一些 4 cell 的 sensor, 往往需要先做QSC等标定，确保pixel 感度差异已经矫正**

gain map 标定通常使用均匀的平板光源即可。可以参考下图
![[attachments/Pasted image 20260526230421.png]]

后文介绍的 CRA mismatch 对PDAF的影像对gain map的需求很大

## DCC map

DCC 标定的目的是将左右像素的离焦量跟调焦电机的位移量对应起来。在镜头中，调焦镜头的位置跟离焦量（或者左右PD的相位差）之间近似线性关系（CRA mismatch 对PDAF的影响中有更完整的实测曲线），DCC 曲线实际就是这个线性关系的 k b。 将PD图分成多个小块，每个小块都拟合一个 DCC k b，就构成了 DCC map。


DCC 标定环境通常是让camera 正对竖条纹图（或者菱形图），这里有两个重要的参数：
1. camera到标定板的距离：一般是调焦镜片位于整个调焦行程中间位置时的对焦距离（即调焦镜片位于最中间时，刚好能看清标定板）
2. 标定板线条密度：
	1. 这里线条不能太宽（可用于左右PD图匹配的信号太少，标定质量下降）
	2. 线条也不能太密，相位差变化超过线条宽度时（此时越过了一个线条），会跟错误的线条匹配（这里可以先用一根线条确认宽度，此外，网上流传的高通PDAF标定文档也给出一个适用于广角镜头的经验公式）
![[attachments/Pasted image 20260526231151.png]]

此外标定 DCC map 时通常可以跟CDAF 算出来的对比度信息判断当前dcc 标定是否正确，如下示意图，蓝色线是DCC 曲线，他的过零点对应PD视差为零，也就是PDAF认为的最清晰的位置，我们拿他跟清晰度计算出来最清晰的位置比较，就知道DCC标定的是否有误差了。
![[attachments/Pasted image 20260526231300.png]]

最后，DCC map 通常会取6 * 8这样的分块大小（分块提升局部标定准确性），且通常标定参数由分块中心到边缘成均匀变化（这是由于系统中场曲存在的影响）

# CRA mismatch 对PDAF的影响

CRA mismatch 对PDAF的影响远比想像的大，那怕这种不匹配在主画面没有明显的影响时，PDAF也是受影响（可能时PDAF的像素采样率更高，因此更敏感）

## PD图像亮度的影响

如下所示，是一篇论文研究 CRA mismatch 对 PDAF 影响时绘制的示意图。其中蓝色时理想CRA时的光线，红色时CRA mismatch 时的结果。以在焦时为例（离焦时亮度的影响是一样的）：
![[attachments/Pasted image 20260522125807.png]]
图片出自：https://arxiv.org/html/2510.27662

1. 对于左PD，图中左侧的pixel原本落在左右PD中心的光线现在向右偏了，这就造成左PD接收的光线减弱，对应PD图更暗。那么右侧呢，看到图中刚好反过来，实际光线像中心偏，左PD接收的光线更多，对应PD图更亮。下图就是此时左PD图像的实拍图（Y方向有裁剪）
![[attachments/Pasted image 20260522125958.png|669]]

2. 对于右PD，其实跟左PD一样，但是照亮的方向刚好相反。参考上面的左PD图像，刚好亮暗反过来，即左侧亮，右侧暗

这里的影响主要表现在以下几点：
1. PD 图亮暗差异对 gain map 的需求更高了，同时噪声分布变化也会影响后续的匹配
2. 被照亮区域更容易过曝，甚至主图像对应区域亮度正常，也会出现 PD 图过曝
3. PD图中心位置会出现剧烈的亮度变化，暗均匀mesh分布的gain map 矫正是无法处理这种情况的，此时只能搞更高精度（非均匀采样）的gain map

## DDC map的影响

对 DCC map 的影响更隐蔽，且不容易定位根因，下面是一个 CRA mismatch 系统中实际标定的DCC 曲线（只取了一行的数据，其他行表现基本类似）。
可以看到这个DDC map 非常诡异：
1. 他的中心两列斜率是正的，且基本满足DCC曲线（镜头实际情况就是类似这样，中心近似线性，超过线性部分开始变弯）。
2. 边缘的两列曲线的线型基本是对的，但是**斜率变为负**
3. 剩下中心跟边缘之间的两列，表现完全异常，标定的曲线也是奇形怪状。看起来就像是**正斜率向负斜率变化时的过度区**

![[attachments/Pasted image 20260522130051.png|697]]

出现这个问题的原因正式CRA mismatch。在一个CRA mismatch的系统中，镜头跟sensor 之间CRA的差异从中心视场向边缘视场变化时，偏差的角度是在逐渐变大的。这意味着：
1. 在中心位置时，CRA 偏差比较小，系统近似正常系统，对应DCC曲线也是正常的
2. 从中心向边缘市场移动时，CRA差异变大，这个差异造成原本聚焦在左PD上的光线向右PD移动（见上面论文中的示意图），同时也会上相邻sensor中原本打在右PD上的光线向自己的左PD移动，此时就会出现光线刚好照在两个大pixel中间，实际进入PD的信号最弱，信噪比最差，同时PD系统假设的左右相位差也完全失效。此时你看左右图像中对应位置，无论离焦程度如何，图像仍处于对其状态。你的DCC map 此处就是一个混乱的状态。显然这里不能用于PDAF对焦。
3. 继续向边缘视场移动，CRA mismatch 的影响已经让原本打在右PD的光进入到左PD了，即左右PD图像的相位变化在这里反转了，你就会看到DCC曲线的极性在边缘区域发生反转。不过这里仍然能够用于PDAF对焦，只需要处理下斜率就行了

# see also

用仿真软件模拟PD工作原理，这里没有sensor:[how-it-works-on-sensor-phase-detect-autofocus](https://blog.reikanfocal.com/2023/05/how-it-works-on-sensor-phase-detect-autofocus/)
