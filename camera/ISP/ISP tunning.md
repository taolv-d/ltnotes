# 1 RAW

DCRAW 源码 [[../../TODO|TODO]]

# 2 ISP turning 原则
## 审美型
**标准：** ISO CIPA CPIQ
**要点：** level + MTF+ noise + color 的平衡
level: 亮度（曝光）、对比度（gamma） 之间是一个矛盾
MTF: shape vs noise （gain, edge roughness, aliasing, overshout, undershout）
noise: 颜色准确性 vs noise (crosstalk, CCM gain, gain)
noise(video): 时域噪声 vs motion blur
## 监控型
特点：信息型+审美型
标准： IEC62676-5
要点： sharpness + noise + artefact
主观指标：看得见、看的准、看的舒服
客观指标：空间分辨率、时间分辨率（运动模糊）、低光（SNR）、低光下缺陷少（SAR，衡量缺陷的指标）、HDR场景（120dB） SNR SAR 要好
亮度 vs 对比度
动态范围（DR） @ SNR1
motion artefact (多帧融合、亮暗背景等)
## 自驾型
特点：信息型 兼顾审美型（给人看的后视镜）MV base + HV base (机器视觉+人类视觉)
IQ Policy: CDP（对比度检测概率） + CSP (颜色区分能力)
标准：P2020
要点：感度、帧率（120Hz）、Motion Artefact Free （安全性）、LED Hickering Mitigation（抗LED闪烁，车灯信号灯等）、Extrame HDR（140dB）、SNR & SNR Drop、畸变、Lens Flare in EHDR（抗眩光）、温度影响
# 3 ISP module 的流程
1、支持哪些module
2、支持哪些格式
3、bypass 支持
4、支持多少input
5、pipline frunction blocks
6、blocks input and output bit depth
7、统计模块（统计的位置，信息、3A、防抖）
**ISP Cmodel**
ISP float model -> ISP fix model -> ISP RTL Model(Asic or FPGA)
ISP float model ->ISP fix/float model -> ISP CPU/GPU
精度控制: 资源控制 vs image quality
**ISP 最小系统**
CFA->BLC->WB->Demosaic->gamma(可选)
# 4 模组差异及IQ对策
差异的来源：lens、sensor、IR、组装、microlens
差异的表现：感度、 dark shading(无光)+pixel shding(有光)+lens shading(LSC) (shading引起模组不线性)
对策：使用golden模组turning, 其他模组通过标定校准到golden模组。使用corner sample 验证IQ turning的结果
corner case 处理：模组厂筛选，放弃掉
要标定的相机的特性：
1、不均匀性：black level in different gain、线性度、shading
2、color: 白点、CCM
3、坏点等
4、色差、几何畸变、LSC
5、Noise profile
6、响应度 responsivity
7、base ISO
8、ISP中的各种系数
LSC 的差异：
矫正方法：mesh、radial shading (省isp，但性能差)
# 5 相机的标定
**需要标定的内容**
black level
LSC
static DPC （坏点）
white blance
CCM
noise profile
AF key position
# 6 绿平衡
目标：消除Gr Gb的不平衡，平坦区域Gr~=Gb,边缘区域不要乱搞
原理：平坦区域Gr Gb 应该相等，但crocess talk 造成他们不相等
矫正：使用周围的G blend 出一个新的G。$G_{cur}=\alpha*G_{avg}+(1-\alpha)*G_{cur}$ 其中$\alpha$根据Gr Gb的差异取不同的值。
前提：需要标定噪声，防止噪声干扰无法区分哪些是边缘，哪些平坦区域
# 7 DCP 坏点矫正
**静态坏点矫正：**
1、sensor 厂 做 wafer test, 保证没有两个相连的坏点
2、模组厂

**坏点检测** 检测邻域差异

**坏点修正** 混合

**turning target** no hot pixel no detail loss
# 8 color correction (看起来简单，实际难)
方法：CCM 3Dlut
前置：色彩空间确定、白平衡（根据色温确定）、BLC、linearized、LSC、正确的曝光、bypass color enhance、gamma(合适的色彩空间)、CSC
目的：sensor RGB->CCM->gamma->sRGB
影响因素：光源（需要高显色指数）、sensor QE 限制上限
# 9 gamma
每种色彩空间都有自己的gamma
gamma 一般分两段：
E709=4.5* L (L<0.018); 1.099 * L^0.45 - 0.099 (0.018 < L)  
# 10 AE
**曝光：**
光的角度：Exposure Value(EV) = lux * sec
camra角度：EV=ISO * shutter * F#

AE 目标：设置ISO(again dgain) shutter F# 适应环境亮度

**两种AE系统**
乘性系统：EV=again * dgain * shutter * F#
加性系统：log2(EV)=log2(F#)+log2(sys_gain)+log2(ET)  老式相机常用

**APEX 曝光系统原理**
Ev = Av + Tv = Bv(测环境亮度，查曝光表) + Sv(胶片感度) （由Bv Sv 推算Av Tv）

AE系统：
RAW->测光模块（light metering, 计算Bv）->AE gain
RAW->场景分类（风光、人像。。）-> AE gain
**light metering** 均值法（分块统计亮度 + 输出直方图）
RAW->64 lum matrix block ->计算每个block的 min max mean->计算相邻block的min max...->分类器->{1、可分类，结合分类结果、AF信息,  2、不可分类} ->生成曝光值->曝光重分配->设置
**classfilter 分类器**
1、逆光（U型直方图，两边高，中间低）
2、运动 亮度矩阵是一个极低分辨率的图像，可以看出运动方向，用于避免运动模糊
4、人物
5、风管、雪景、。。。corcase

**AE 均值法**
中性灰（122）经过反gamma（2.2）后的亮度是 56 （全图平均值位56），对应sRGB
Evbase, Evshift 给AEtaeget 乘一个系数来调整
Exposure convergence 收敛速度（vs 稳定性）
 需要限制 最大最下shutter、again dgain ispgain sysgain (sysgain = again * dgain * ispgain)
 曝光分配， AE Route：线性 阶梯 混合
 Anti flicker(抗频闪)：1、限制最小曝光（1/50 s），可能过曝。2、调整为半周期整数倍（避免过曝）
# 11 AWB
### AWB统计
**AWB目标：** 对于灰色，R=G=B
**实现方法：** 对 RGB 乘gain 调整 R/G B/G
**影响AWB的因素：** R G B 的线性（全图global的线性及local的线性），不线性表现在1、ob不同，K不同，或者曲线弯了
**AWB统计的位置：** 线性后，DPC(去掉OB)LSC等后面
**AWB统计的方法:** 
	R G B 的mean —— 灰度世界法
	R G B 的直方图 —— 完美反射法
	R G B 分区域的直方图和均值 —— 加权白块法
**AWB统计的参数：** 
	统计的ROI, 例如避开边缘区域
	统计分块的数量，分块越密越能检测到小的白点，但稳定性会随之下降
	权重，控制哪些块参与统计，或完全不要（权重=0，如遮挡、语言边缘黑色区域）
	可靠点数量统计、异常点检测
	filter 例如过亮过暗的不要，只要中间线性的部分（bl--wl之间）
	利用 b/g r/g剔除异常点，同时也可以用于判断环境颜色
## AWB algorithm
**目标：** 输出 WB gain, CCT（色温）， 场景分类（可选）
**input:** AWB 统计、brightness、AF统计（区分纹理、平滑区域等）
**AWB 流程**：
	1、initial calibration，需要对模组标定一个曲线（光源色温与sensor相应关系），如下图
	2、估计CCT, narrow down CCT：把实际光源定位到标定曲线上，同时排除干扰项
	3、场景分类（结合多维信息，难点场景通常时肤色、浅蓝、浅棕等）
	4、白点加权，多个光源加权（如同时有室内跟室外）
	5、计算白点，人对偏蓝会不适，不要偏蓝
	6、color apperance：根据视觉非恒常性，调整2500k--7500k 以外的光源：
		2500k--7500k 正常
		<2500k 偏黄一点
		>7500k 偏蓝一点
![[attachments/Pasted image 20260512223235.png]]
**AWB算法选取：** 考虑算例、是否参考历史帧等
## AWB turning
**先决条件：**
	1、gloden 模组
	2、lens IR 正常
	3、线性化部分已OK(blc wdr)
	4、shading 校准（尤其color shading）
	5、AE 调好， AWB 会参考AE
	6、bypass 所有影响颜色的模块
	7、AWB initial calibration 完成
	8、AWB 统计完成

# 12AF

## AF算法

**目标:** 计算lens在不同位置的清晰度FV, 找到峰值位置

**方法：** 

​	梯度法：sobel 拉普拉斯算子等，找到梯度最大的位置

​	高频分量法：使用高滤波器（如下的流程）
![[attachments/Pasted image 20260512223319.png]]
**image 预处理：** ROI crop、bayer2gray (加权法、抽G) 减blc、gamma提亮

**pre filter：** 去椒盐（中值）

**focus filter:** 通常是二维可分离滤波器（高通，水平、垂直、窄带、宽带组合成4个，宽窄对应亮暗场景）

​	通常水平 IIR滤波器，竖直是FIR滤波器（使用IIR需要buffer，资源消耗大）

**LDG:** 抑制光源对FV的影响（lens不同位置时，点光源的大小会变，模糊区域，点光源变大，低频变多，造成FV高。但是AF 希望整个场景一致，清晰度不一样） 

​	抑制点光源原理：根据亮度衰减滤波器输出，收缩光晕。只统计高亮区域

**coring:** 提升暗光性能（低光噪声大，需要提升FV曲线的抗噪性能），使用双阈值，太黑的不要，饱和的设为固定值，中间区域的线性映射

## AF algorithm (3A里面最难的)

**对焦分类：**

1、主动对焦：利用激光测距

2、被动对焦：

​	cdaf: 移动镜头找最清晰

​	pdaf: sensor 内置相差传感器，根据相差与距离的查找表找到对焦位置

**对到哪：** 场景识别 + 多点对焦

**何时refocus:** 

​	跟踪（运动估计、场景分析）

​	场景中有其他东西进入（分割前后景，看前后景变化，例如有汽车进入）

**AF流程：**

1、场景不变，则loop

2、选择 scence mode

3、选择 region, 确定对到哪

4、FV filter：调整AF 统计的filter

5、apply weight: 根据mode scence 重设权重

6、Do PDAF: 速度快，但精度不高

7、Do CDAF: PDAF之后再精确对焦

**AF 曲线**

通过调整统计部分的参数，调整曲线的类型。曲线太缓不容易找到最清晰的位置。太尖不好用三点法找到峰值

也不是在最尖的位置最好，最尖的地方清晰度宽容度差一些，不在焦面的会很糊
![[attachments/Pasted image 20260512223342.png]]

**根据scence动态调整策略**

scence分类：前景、背景有无人脸、运动场景、背景有纹理、低光、点光源、低对比度等

# 13chroma adjust 色调调整
在YUV域色度与亮度分离，单独操纵色度
用处：1、不同ISO下控制饱和度（ISO高时高饱和，低时低饱和，实现降彩噪）2、不同亮度下控制饱和度（pixel亮时高饱和，低时低饱和，实现降彩噪）
![[attachments/Pasted image 20260512223354.png]]
# 14sharpening
**edge** 强边缘，有方向
**detial** 重复的、对比度不高的纹理，无方向性。如头发、毛发、毛线球等
**流程** 输入YUV图像，经过filter 后得到edge，detial, base(低频). 分别对edge 和 detial 进行加强。最后对各个结果(包含输入图像)进行blend
**shoot ctrl**: 效果见下图
![[attachments/Pasted image 20260512223413.png]]
# 15 WDR
## WDR sensor
实现方案：
**时域WDR：**
frame base 多帧加权求均值。前后两帧曝光不同
line base 以行为基础，每一行曝光两次（监控常用），比frame base 两帧间隔要小
缺点：
	1、引发假边缘（SNR 不同引起）
	2、banding: 光源有波动性，造成需要更长的曝光
	3、LED flicker （LED PWM调光引起闪烁）
**空域WDR:** 大小像素的感度差异
**DCG:** 大小像素 + 时域多帧
**Lofic：** sensor 中集成一个电容，用来存储电荷，但电容容易受温度影响
## WDR pipline
linear -> 图像最稳定
WDR -> 90-120dB
hybrid -> 自动切换
如何处理运动：
1、不处理，直接加权求和
2、检测+融合：
	检测：利用公式 $I_长/R -I_短>T$, T需要根据noise 水平来定
	融合：短曝噪声高，长曝有运动模糊。一般小的用长曝，大的用短曝，中间部分混合，各个结点需要根据noise水平来确定
	可能存在的问题：1、可能短曝的有频闪，长曝没有。2、噪声问题如何改善（例如强制用长曝）
sensor 端压缩（节省传输带宽，如20bit->16bit）,实现方法：分段等比例压缩，结点要根据噪声来考虑
## WDR对哪些模块有影响
1、initial calibration AWB，shading, CCM 无影响。noise 有影响
2、AE 以及 AWB中受AE(主要是照度)影响的部分
3、Tone mapping 影响很大
4、demosic 无影响
5、CCM 不需要调，受AE影响的部分（饱和度）需要调
6、gamma 静态不需要调，动态类似 tone mapping
# 16 3D 降噪
**目标：** 去除时域噪声，同时保持运动物体的可见性
**对象：** 通常Y通道
**时域噪声：** 不想关噪声，n帧平均，时域噪声降低n^0.5倍，证明如下
$$\begin{equation}\begin{aligned}
两个图像相加: S=S_1+S_2 \\ 
时域噪声：\sigma^2_t=\sigma^2_{t1}+\sigma^2_{t2} \\
s/\sigma_t=(s_1+s_2)/(\sigma^2_{t1}+\sigma^2_{t2})^{0.5}\\
当\sigma^2_{t1}=\sigma^2_{t2} ，s_1=s_2时\\
s/\sigma=2^(0.5)*(S_1/\sigma^2_{t1})\\
对于n帧,则s/\sigma=n^(0.5)*(S_1/\sigma^2_{t1})
\end{aligned}\end{equation}$$
**流程：** Y->2D空域降噪->帧间运动分析，前后景分割->3D降噪（空域时域，多级）
**2D空域降噪：** 双边滤波，BM3D, none local mean等（基于搜索相似快的方法）
**运动检测（关键）：** 比较块的变化。分割前后景、得到运动矢量