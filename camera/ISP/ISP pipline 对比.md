本文记录了一些互联网上能找到的商业ISP pipline，总结ISP的几点作用：
- ISP 修复了camera 的缺陷：
	- 坏点矫正：修复坏点
	- AWB, CCM 等修正camera色彩响应与人眼不一致
	- CAC 矫正紫边问题
	- LSC 矫正shading
	- demosaic 修复bayer采样造成的数据丢失
	- ...
- ISP 实现了非标转标的过程：camera直出的图像属于设备相关色彩空间，经过ISP(WB、CCM、gamma等)转为标准色彩空间
- ISP 对数据进行了压缩：ISP 输入通常是10/12bit(HDR模式下更高)，经过ISP后输出变为8bit YUV图像，本质压缩了荣誉信息（噪声也被压缩了）

ISP 中很多模块的位置不是固定的，但是 demosaic、CSC这两个模块比较关键，它们将ISP分为3个域: raw、rgb、yuv。通常BLC是第一个模块，他用来矫正黑点平。

ISP的一个很大的任务就是降噪，降噪通常在raw 或者 yuv 域进行。
- raw 域噪声通常还是比较容易建模的泊松-高斯噪声，在源头降噪收益很大。
- yuv 域亮度和色度分离，可以分别针对彩噪、亮度噪声执行不同降噪手段（UV域猛降彩噪，不会造成画面模糊）
- rgb 域 降噪属于**费力不讨好**，相比raw 跟yuv，这里计算量更多，同时三个通道耦合，噪声分布也复杂化了

当前列举的这些ISP pipline 应该是近几年的设计，可以看到当前应用的ISP基本都是3D降噪。不过现在越来越多nn方法的ISP概念已经提出。目前看到的理论：
	raw图进ISP之间先过一个nn 降噪网络（或者ISP中间的某个位置下DDR执行nn降噪）。
	AWB / AE 使用nn 方法估计参数
	3D降噪使用 nn 模型做动静判断
	对比度增强等使用nn 方法生成曲线或者直接nn出图


简化的参考ISP
![[attachments/Pasted image 20260516093318.png]]


**RK ISP39**
图片来自 [RK3576 MIPI Camera ISP调试：主观调优与工程实战（下） - myfeiyang - 博客园](https://www.cnblogs.com/cbd7788/p/20044639)

![[attachments/Pasted image 20260519141424.png]]

- localSgmStrg:
	- 这个ISP pipline 中虚线部分很有意思，他串起了几个降噪模块跟lsc/drc这些会修改图像亮度分布的模块，本质是再记录图像不同区域经过ISP处理后噪声水平提高或降低的程度。例如，经过LSC后图像边缘被拉亮，噪声水平提高，再后续的降噪模块中，边缘区域可以给更强的降噪力度来平衡。
- btnr:
	- 这也是这个ISP中比较有意思的部分，他是一个结合时域跟空域的降噪模块，看起来这里是在做3D降噪。通过动静判别（猜测是基于块亮度实现），对于静止区域叠加（直接在raw上叠加？还是插值后的r\g\b三个图叠加？）。同时这里叠加后噪声分布已经变化，通过localSgmStrg 传递到后续模块（如sharpen 补偿清晰度损失）

**Hi3559V200** 
图片来源：[海思Hi3559V200的ISP流程分析-CSDN博客](https://blog.csdn.net/feifei126/article/details/137081531)

![[attachments/Pasted image 20260519175858.png]]

**RDK X5**
资料来源：[5.17.7.1. 修订记录 — X5 芯片用户手册 1.1.2 文档](http://1.12.36.124/multimedia_development/isp_tuning_guide/isp_tuning/isp_tuning.html#isp-tuning-overview)
![[attachments/Pasted image 20260519144129.png|697]]
RDK文档开放比较多，这里以它为例简介模块作用：

| 节点                                        | 域            | 主要作用                                         | 关注                                                          |
| ----------------------------------------- | ------------ | -------------------------------------------- | ----------------------------------------------------------- |
| Sensor Input                              | RAW          | sensor 输出 Bayer / RGBIR / HDR 多曝光 RAW        | 先确认分辨率、帧率、bit depth、Bayer pattern、HDR 曝光比、gain/exposure 范围  |
| HDR stitch                                | RAW / HDR    | 长短曝光帧融合，提高动态范围                               | 含 Deghost，支持多曝光运动去鬼影                                        |
| AE Statistics / AE                        | 统计/控制        | 自动曝光，控制曝光时间、模拟增益、数字增益、目标亮度                   | 分块统计、ROI 测光、背光补偿、低光 target 压缩、防频闪、曝光分解                      |
| AWB Statistics / AWB                      | RAW 统计/控制    | 自动白平衡，估计色温并输出 R/B gain                       | 分块统计，支持光源概率、纯色校正、色温偏好、ROI AWB、Face AWB                      |
| AF Statistics / AF                        | 统计/控制        | 自动对焦                                         | 支持 CDAF、PDAF、Hybrid；                                        |
| TPG                                       | RAW          | 生成测试图                                        |                                                             |
| RGBIR                                     | RAW 前端，可选    | 处理 RGBIR sensor，把 RGBIR 转成普通 Bayer 并提取/去除 IR | 内部像小 pipeline：BLS、DPCC、IR 通道降噪/升采样、G/R/B 插值、IR remove       |
| Expand                                    | RAW / HDR    | 数据解压缩/压缩，常用于 WDR/PWL sensor                  | Expand 可把 12-20 bit 输入扩到 20 bit；Compress 再压回输出 bit depth    |
| BLC                                       | RAW 前端       | 黑电平扣除                                        | 黑电平可随 gain 标定                                               |
| Green Equalization, GE                    | RAW          | 修正 Gr/Gb 不平衡                                 | 主要减少 demosaic 后的棋盘格、pattern、false color                     |
| Defect Pixel Cluster Correction, DPCC/DPC | RAW          | 坏点/亮点/坏点簇校正                                  | 支持静态坏点表和动态检测                                                |
| 3DNR                                      | 时域/RAW 或中间域  | 时域降噪                                         | 基于帧间运动检测、motion mask、motion dilation；要防拖影                   |
| 2DNR                                      | RAW          | 空间降噪                                         | 使用 NLM，分高/中/低三层；支持 luma control、LSC compensation、运动/静止增强    |
| Lens Shading Correction, LSC              | RAW          | 修正镜头暗角和颜色不均匀                                 | 按空间 mesh 做校正，Bayer 四通道独立；至少标定高/中/低色温表                       |
| Digital Gain                              | RAW          | RAW 域数字增益                                    | 可分 R/Gr/Gb/B 通道设置，用于亮度和通道校正                                 |
| WDR                                       | RAW/亮度映射     | 宽动态压缩和局部/全局对比度增强                             | 以 16x16 block 做局部统计；含 low/high/global strength、局部曲线、halo 抑制 |
| Demosaic                                  | RAW -> RGB   | Bayer 插值成 RGB                                | 不只是插值，还处理 CAC、Sharpen、黑白边、紫边、伪色、摩尔纹                         |
| De-Purple / DeFalse Color / Demoire       | Demosaic 子模块 | 抑制紫边、伪色、摩尔纹                                  | 和锐化、插值、GE、DPCC 强耦合                                          |
| CAC                                       |              | 色差校正                                         | 以 G 平面为基准缩放 R/B 平面，通常由 Calibration Tool 标定，不建议手改            |
| CCM                                       | RGB          | 颜色校正矩阵                                       | 3x3/3x4 矩阵校正颜色串扰和色彩偏移；可按色温/gain 做多组参数                       |
| Gamma                                     | RGB          | 非线性亮度映射、对比度调整                                | 64 节点 LUT；改 Gamma 会影响颜色指标，可能需要重新验证 CCM                      |
| Edge Enhancement, EE                      | Y/RGB 后端     | 边缘增强、清晰度提升                                   | 区分边缘和细节；gain 高时通常降低强度，避免噪声和过锐化                              |
| CSC                                       | YUV          | 色彩空间转换                                       |                                                             |
| Color Processing, CPROC                   | YUV          | 亮度、对比度、饱和度、色调调整                              | 后置风格化模块，通常 Phase 3 按应用偏好和 gain 微调                           |
| CNR                                       | YUV          | 色度降噪                                         | 三层金字塔模型处理彩噪；太强会引入红/蓝偏色或颜色发灰                                 |
| CROP                                      | YUV          | 裁剪                                           |                                                             |
| Output                                    | YUV/RGB      | 输出                                           |                                                             |

号称安霸ISP pipline（真实性存疑）
图片来源：[简述安霸pipeline及其关键参数--raw域模块](https://www.shuzhiduo.com/A/amd0QeVmzg/)
![[attachments/Pasted image 20260519191013.png]]
这里mctf看起来是多帧降噪