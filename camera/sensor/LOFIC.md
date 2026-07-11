---
type: note
status: done
tags:
  - hdr
  - lofic
rating: 0
create: 2026-07-11
update:
---
**LOFIC** （Lateral Overflow Interation Capacitor， 横向溢出集合电容）的思想是，光照太强时产生的光电子超过满阱容量时，溢出的电荷被一个专用的电容收集，而不是溢出到周围像素中。下图中Cs就是用于收集溢出电荷的。
LOFIC 目的是增加动态范围。限制动态范围的两个因素是：1 本底噪声，2 满阱容量。噪声这里的控制已经很难在提升，因此Lofic技术是进一步提升动态范围的主力。（更进一步提升则是TTS）

![[attachments/Pasted image 20260711232520.png|321]]

![[attachments/Pasted image 20260711232537.png|466]]
LOFIC的工作时序参考下图（N2 对应强光，N1 对应弱光）：
- t1 复位：R S T 导通，Cs Cfd 被复位
- t2 采样复位噪声 N2：S 打开，R T关闭，采样为 Cs Cfd 的复位噪声
- t3 积分：弱光时仅PD 内有电荷，强光时电荷溢出，依次进入Cfd Cs
- t4 采样复位噪声N1：R S T均关闭，采样为Cfd 复位噪声（如果溢出，此处采集的信息实际无效了）
- t5：T 打开，PD中电荷转移到FD中
- t6 采样 S1 + N1：T S 关闭，此时采样 S1 + N1
- t7 采样 S2+N2: S 打开，此时采样S2 + N2

可以看到：
1. **HCG**：t4~t6 这三步操作对应的是暗光场景，没有出现电荷溢出时的情况。跟传统的CIS CDS 流程一致。不过如果是强光，这里采集到的信号经过CDS 后应该变成0了。
2. **LCG**：t2 t7 两步对应的是强光场景，此时电容是Cfd+Cs。这里的问题是噪声跟带噪信号之间间隔太长了，因此这里噪声不具备相关性。这里减噪声操作被称为DDS（差分双采样）。DDS 抑制噪声能力弱很多，这也是后续改进的主要方向。

另外，LCG 转换这里用了更大的电容，那么转换增益会变小，后级用同一个ADC 采样，对应这里的分辨率势必会更低。不过这刚好与人眼特性匹配上了（人对亮度的变化是波动超过1%才能察觉，对应亮区需要亮度变化很大人眼才能感知，参考[[../optics/color/影视从业者对色彩的理解：Color science and digital image|影视从业者对色彩的理解：Color science and digital image]]）。因此LOFIC这里降低分辨率实际影响并不大。