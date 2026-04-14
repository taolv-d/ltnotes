## 这个仓库把传统 ISP 拆成几段“可学习但可控”的模块

输入读取 -> raw 预处理 -> raw denoising -> AWB/CCM color correction -> linear sRGB -> photofinishing -> guided upsampling -> enhancement -> sharpening -> JPEG保存/可选raw回嵌

**1. 输入阶段：先统一成 raw + metadata**  
在 demo.py (line 264) 里，输入支持三类：

- DNG
- 16-bit PNG raw + JSON metadata
- JPEG/PNG，如果是他们自己保存过的 JPEG，里面可能嵌了 raw 和 metadata；否则会走线性化

具体分支是：

- DNG：extract_image_from_dng -> normalize_raw -> demosaice
- PNG-16：直接读 raw，同时找配套 JSON
- 普通 JPEG/PNG：先 net.read_image() 试着从 JPEG 尾部 payload 恢复 raw；恢复不到就加载线性化模型，把 sRGB 近似“反 ISP”回 raw 空间


**2. raw 预处理**  
DNG 进来以后会先做最基本的 sensor 域处理：

- 黑电平/白电平归一化：normalize_raw(...)
- Bayer CFA demosaic：demosaice(...)

这里仓库没有把“demosaic”学进去，而是把它当成输入准备的一部分。进入主网络时，raw 已经是 3 通道线性 raw RGB 了，不是 Bayer mosaic 了。

**3. raw denoising** 
[[../../machine learning/rgb_增强网络/NAFNet|NAFNet]]
进入 PipeLine.forward (line 739) 后，第一段主干是 raw 去噪：

- S24 相机可走 camera-specific denoiser
- 其他相机走 generic denoiser
- 网络骨干是 NAFNet

代码里这一步输出 denoised_raw，而且支持强度插值：

**denoised_raw = (1-a) * raw + a * denoised_raw**

也就是说，去噪不是硬开关，而是连续可调。

**4. AWB + CCM：从 raw RGB 变到 linear sRGB**  
这是仓库最 ISP 味儿的一段，在 _color_correction(...) (line 553)。

它要解决两件事：

- illum：白平衡光源颜色
- ccm：颜色校正矩阵

流程大致是：

- 如果用户没直接给 illum
    - S24 图像可用 S24 专用 AWB 模型
    - 其他图像用 cross-camera AWB 模型
- 如果用户指定了 target_cct / target_tint
    - 会把 illuminant 映射到目标色温/色调
- 如果 metadata 足够完整
    - 用 forward_matrix/color_matrix/calibration_illuminant 动态算 CCM
- 否则退化成 metadata 里的固定 color_matrix

最后真正做颜色映射的是 raw_to_lsrgb(...) (line 292)：

- 先按 illuminant 算白平衡增益
- 再乘 CCM
- 得到 lsrgb，也就是 linear sRGB

所以这里很关键的一点是：**photofinishing 之前，图像已经不在 raw 相机色彩空间，而是在 linear sRGB 里了。**

**5. 可选的曝光和轻量降噪**  
在 lsrgb 上，还会有几步可选处理：

- auto_exposure
- 手动 ev_scale
- 非学习式 luma/chroma denoising

也就是说，仓库把“raw denoiser”和“lsrgb 上的亮度/色度平滑”分成了两层，不混在一起。

**6. photofinishing：仓库最核心的“风格渲染模块”**  

它不是一个直接出图的大黑盒，而是顺序预测并应用一组 ISP 风格参数，执行顺序是：

1. GainNet：预测数字增益
2. GlobalToneMappingNet：预测全局 tone mapping 参数
3. LocalToneMappingNet：预测局部 tone mapping 系数
4. LuTNet：预测二维 CbCr chroma LUT
5. GammaNet：预测 gamma

对应 forward 顺序在 photofinishing_model.py (line 825) 非常清楚：

lsrgb -> gain -> GTM -> LTM -> chroma mapping -> gamma -> srgb

这里它的设计特别有意思：

- 亮度和色彩分开处理
- GTM/LTM 负责 tone
- CbCr LUT 负责色彩风格
- gamma 放在最后做 display-referred 输出

所以这个模块本质上是在学一套“神经版、可解释的 photofinishing operator stack”。

**7. 为什么 photofinishing 先下采样**  
默认会先把 lsrgb 缩小到 1/4 再跑 photofinishing，然后用 BGU 上采样回来：

- 下采样是为了省算力，也让全局/局部 tone mapping 在低分辨率上更稳定
- 上采样用的是 bilateral_guided_upsampling.py (line 1)

也就是：

- 低分辨率图上学风格变换
- 高分辨率图保细节
- 用引导上采样把两者结合起来

这一步是这个仓库“又快又尽量保边缘”的关键工程点。

**8. enhancement：最后再做细节增强**  
在 photofinishing 完成后，若加载了 enhancement 模型，会再跑一个 NAFNet 做细节增强。

顺序上它是在 photofinishing 之后，而不是之前。含义是：

- 先把色彩、tone、风格定下来
- 再补局部细节和质感

这比把 enhancement 混进 tone mapping 更容易控制，也更符合 photo finishing 的工作流。

**9. sharpening 和输出**  
最后还有两层输出侧处理：

- 可选 sharpening
- 可选 apply_orientation

最终输出 srgb。如果用户用 store_raw，则 save_image(...) (line 1419) 会把：

- 最终 JPEG
- 压缩后的 raw
- metadata
- 编辑参数

一起塞到 JPEG 尾部 payload 里。这样后面再打开这张 JPEG，还能把 raw 恢复出来重新渲染。这就是它“可无限重编辑”的实现基础。

**一句话总结**  
这个仓库的完整 pipeline 其实可以概括成：

- 先把任意输入尽量还原到统一的 raw 表示
- 在 raw 域做去噪
- 用 AWB + CCM 进到 linear sRGB
- 用一个“分阶段、可解释”的 photofinishing 模块完成风格渲染
- 再做细节增强和输出封装

所以它本质上是一个“模块化、可编辑、可重渲染”的 neural ISP，而不是单个 end-to-end raw2rgb 网络。


## 本文复用了现有的工作
- denoising 复用了 NAFNet
- enhancement 也复用了 NAFNet
- upsampling 复用了经典的 BGU 思路
- linearization 明确写了是基于已有的 CIE XYZ Net


## PhotofinishingModule 里有 5 个主分支

- GainNet：预测数字增益
- GlobalToneMappingNet：预测全局 tone mapping 参数
- LocalToneMappingNet：预测局部 tone mapping 系数图
- LuTNet：预测色彩 CbCr 二维 LUT
- GammaNet：预测 gamma

**GainNet 和 GammaNet：最简单的两个头**  
这两个结构几乎是同一套模板，底层都来自 BaseNet：

- 几层卷积
- 中间插一个 MultiBranchConvBlock [[../../machine learning/nn积木/MultiBranchConvBlock|MultiBranchConvBlock]]
- 再插一个 CoordinateAttention [[../../machine learning/nn积木/CoordinateAttention|CoordinateAttention]]
- 最后全局池化到 1x1
- Linear -> Sigmoid

所以输出其实是一个全局标量（比例，需要映射到gain_min~gain max 或 gamma min~gamma max）。


**GlobalToneMappingNet：预测全局 tone mapping 参数**  

它比 BaseNet 稍复杂一点，但思想类似：

- 输入先 resize 到固定尺寸
- Conv + GroupNorm + Act (Conv 看内容，Norm 稳训练，Act 加表达力)
- MultiBranchConvBlock
- CoordinateAttention
- 再几层卷积和池化
- 最后 Linear(…, 3) + Softplus [[../../machine learning/nn积木/Softplus|Softplus]] 确保输出为正且平滑

最终输出 3 个正数参数 gtm_params。

gtm 曲线： `T(x; a,b,c) = x^a / ( x^a + (c(1-x))^b )`
- x 是输入像素值，范围基本在 (0,1)
- a, b, c 是 GTM 网络预测出来的 3 个正参数
- 输出 T(x) 也在 0~1 之间

**这条曲线长什么样**  
是一个 S 型/压缩型的归一化非线性映射。  
它有几个很好的性质：

- x 趋近 0 时，输出趋近 0
- x 趋近 1 时，输出趋近 1
- 中间部分的弯曲程度由 a,b,c 控制
- 能自然做高光压缩和中间调重分配

**3 个参数各自大概在干什么**  
直观上可以这么理解：

- a：控制暗部一侧 x^a 的形状
- b：控制亮部对抗项 (c(1-x))^b 的形状
- c：控制两边平衡点，影响曲线整体偏移/压缩强度

虽然三者是耦合的，不是完全独立，但大致可以这么想：

- a 变大：暗部响应会更“硬”
- b 变大：亮部端压缩方式会变
- c 变大：会改变整条曲线的中心和平衡位置

**它不是按亮度单独做，而是对 RGB 每个通道直接做**  

- GTM：全图统一的三参数 tone curve
- LTM：每个局部位置自己的三参数 tone curve，再和 GTM 结果做混合

**LocalToneMappingNet：最复杂、最核心的一块**  

它不是直接输出像素值，而是输出局部 tone mapping 系数图 ltm_params。整体分两部分：

- guide 分支：生成引导图
- grid 分支：生成 bilateral grid 系数
- 最后通过 bilateral_slice 从 grid 取出每个像素的局部系数

这其实非常像“神经化的 bilateral tone operator”。

先看 guide 分支。它内部用了 MultiScaleGuidanceNet，在 photofinishing_model.py (line 242)。
MultiScaleGuidanceNet 本文提出结构：
这个网络会取同一输入的三种尺度：

- x_low = avg_pool(x, 4)
- x_mid = avg_pool(x, 2)
- x_high = x

每个尺度各自过一个 _guide_conv() 小网络，再上采样回原尺寸，最后拼接融合成单通道 guide。这个 guide 决定 bilateral slice 的深度坐标，相当于“这个像素应该落在哪个 tone bin 里”。

再看 grid 分支：

- 输入不是单张图，而是 concat([x, x_gtm])
- 其中 x 是 gain 后图，x_gtm 是 global tone 后图
- 所以 LTM 明确建立在 GTM 之后，是在“全局 tone 已经定下来的前提上”再做局部修正

grid 分支结构里有：

- Conv + GroupNorm + Act
- MultiBranchConvBlock
- CoordinateAttention
- 再几层卷积
- AdaptiveAvgPool2d(grid_size, grid_size)
- 1x1 conv
- Softplus

最后输出 reshape 成：

[B, num_coeffs, grid_depth, grid_size, grid_size]

然后用 _bilateral_slice(...) photofinishing_model.py (line 426) 按 guide 图对 3D grid 做采样，得到每个像素的局部 tone 系数。

这里很重要的一点是：**LTM 预测的不是颜色，而是局部 tone 参数场**。这就是它为什么比普通 CNN 渲染头更可解释。

另外它还有一个很工程化也很关键的选项：post_process_ltm=True。

开启时它会：

- 在多个尺度上分别预测 LTM 系数
- 上采样后取平均
- 再用 _bilateral_solver(...) 做 refinement

目的就是 README 里说的减 halo。这部分非常像在把神经网络输出重新投影成一个更平滑、边缘一致的局部 tone 场。


ltm gtm 这里的图像变化：（x为gain后图像）

`x --GTM--> x_gtm x + x_gtm --LTM网络--> 局部参数 x --局部tone公式--> x_ltm 输出 = (1-w) * x_gtm + w * x_ltm`

所以 GTM 是 base，LTM 是在它基础上做局部偏移。

**一句更完整的表述**  
我建议你把这块记成下面这句话，基本就很准了：

- GTM 先给 gain 后图一个全局 tone 结果 x_gtm
- LTM 再根据 x 和 x_gtm 预测一个 bilateral grid 参数场
- 每个 grid cell 里存一组局部 tone 参数 w/a/b/c/g
- 每个 pixel 通过多尺度亮度引导生成的 guide，从这个 grid 中插值取出自己的局部参数
- 然后把局部 tone 结果 x_ltm 和全局 tone 结果 x_gtm 按权重 w 混合

五个参数：
- a/b/c：调镜头前面的那条曲线
- w控制ltm gtm 混合比例
- g：先把输入往左/往右推一下，再进这条曲线

所以在局部 tone mapping 里，g 很适合做：

- 某块区域稍微提一点局部曝光
- 或者稍微压一点局部亮度
- 让后面的 tone curve 在不同区域表现出不同响应


**6. LuTNet：色彩映射头**  
LuTNet 在 photofinishing_model.py (line 103)。

它做的不是 RGB 空间直接回归，而是：

- 把图像转到 YCbCr
- 只对 CbCr 做映射
- 输出一个二维色彩 LUT

输入是 ycbcr，先 resize 到固定大小，然后分成：

- y：亮度
- cbcr：色度

这里有两个关键子思路。

第一，基于色度分布建模：  
它对 CbCr 计算可微分二维 histogram，函数在 photofinishing_model.py (line 205)。

然后把这个 histogram 和 identity grid 拼起来，作为主干输入。意思是：LUT 预测不仅依赖当前颜色坐标，还依赖“整张图整体有哪些颜色”。

第二，亮度引导：  
它单独用 _y_net 从亮度 Y 里提一个 attention 向量，再去调制主干特征。意思是：同样的色彩风格 LUT，会根据当前图像亮度结构适配。

网络大致流程：

- hist_net 从 histogram 提取特征
- 拼上 identity CbCr grid
- 编码器若干层
- CoordinateAttention
- 用亮度分支 y_net 做通道重标定
- 解码器输出 2 通道偏移
- 加到 base LUT 上，得到最终 LUT

所以 LuTNet 学的是“在 identity CbCr map 基础上怎么偏移”，而不是凭空生成一个完全自由的 LUT。这会更稳。

## 训练

**photofinishing 这部分是整模块端到端联合训练的**，但训练时对中间阶段加了显式约束，所以它不是“纯黑盒联合训练”。

最关键的证据在 train.py (line 181)：

- 直接 optimizer = Adam(model.parameters(), ...)
- model 就是整个 PhotofinishingModule
- 一次前向 out_images = model(in_images, training_mode=True)
- 一次总 loss 回传 loss.backward()

这说明：

- GainNet
- GTM
- LTM
- LuTNet
- GammaNet

训练对是：

- 输入：denoised raw 图，先用 metadata 里的 illum + ccm 转成 linear sRGB
- GT：目标风格的最终 sRGB 图

然后 loss 不是只看最终输出，而是多项联合，在 loss_utils.py (line 163)：

- 最终图像的 L1
- SSIM
- VGG perceptual
- Delta E
- CbCr loss
- LUT smoothness TV
- LTM smoothness TV
- TM loss
- luma energy consistency

**这意味着什么**  
意味着虽然网络是联合训练的，但作者在 loss 里给了中间结构“扶手”：

- tm_loss 约束 GTM/LTM 的亮度映射别乱来
- cbcr_loss 约束色彩映射部分
- lut_smoothness 让 LUT 更平滑
- ltm_smoothness 让局部 tone 参数图别太抖
- luma_energy_consistency 让 tone mapping 不至于把整图亮度搞崩

所以它的训练范式可以概括成：


那怎么反向传播呢？中间很多lut曲线变化的部分？

8:17 PM

可以正常反传，因为这套 photofinishing 里的中间“曲线/LUT 变化”基本都写成了**可微分操作**。

核心原则就是：  
**网络不直接学离散查表，而是学可微分的参数化映射和可微分采样。**

你可以按几块看。

**1. GTM / LTM 的曲线是可微的**  
GTM 和 LTM 用的 tone mapping 公式是：

text

`T(x) = x^a / (x^a + (c(1-x))^b)`

这里面虽然看起来像“曲线”，但本质只是：

- pow
- add
- mul
- div

这些都是 PyTorch 自动求导支持的。  
所以 loss 对输出的梯度可以一路传回：

- 图像输入 x
- 曲线参数 a,b,c
- 再继续传回预测它们的 GTM/LTM 网络

LTM 里的 w、g 也一样：

- x = x * g
- out = (1-w) * x_gtm + w * x_ltm

这也是标准可微运算。

**2. 2D LUT 不是“硬查表”，而是双线性采样**  
LuTNet 的关键不是生成一个 LUT 就结束，而是后面应用 LUT 的方式在 _apply_2d_lut_on_cbcr (line 1188)：

python

`out = F.grid_sample(lut2d, grid, mode="bilinear", align_corners=True)`

这很关键。

如果是传统离散 LUT 查表：

- 找最近格子
- 直接取值

那会不可导或梯度很差。

但这里用的是 grid_sample：

- 对 LUT 网格做双线性插值
- 对采样坐标可导
- 对 LUT 本身也可导

所以梯度既能回到：

- lut2d 的每个格点值
- 也能回到 grid，也就是 CbCr 坐标

再进一步回到：

- LuTNet 参数
- 前面图像特征

**3. LTM 的 bilateral slice 也是可微采样**  
LTM 里从 bilateral grid 取每像素参数，也是靠 grid_sample 完成的，在 _bilateral_slice (line 426)。

所以逻辑和 LUT 很像：

- 不是“某像素硬选一个格子”
- 而是对 3D grid 做连续插值采样

这就保证了梯度能反传到：

- grid 分支输出的系数体
- guide 分支输出的引导坐标

所以 guide 也能学到“怎么把不同像素送到合适的 grid 深度位置”。

**4. 直方图也做成了可微版本**  
LuTNet 里连 histogram 都不是传统硬 binning，而是高斯软分桶，在 _differentiable_cbcr_histogram (line 205)。

它做的是类似：

- 每个像素的 CbCr
- 对附近多个 bin 都赋一个连续权重
- 用 exp(-dist^2 / 2sigma^2) 做软计数

这样 histogram 也是可导的。  
否则如果是硬计数，梯度几乎没法回传。

**5. gamma 也可微**  
gamma 部分就是：

python

`x_srgb = torch.pow(x, gamma_factor)`

只要加了 clamp 避免 0 和数值炸掉，照样可导。

**6. 作者专门做了数值稳定处理**  
你注意到他们写了 safe_pow(...)、clamp(EPS, ...)，就是为了让这些“曲线型操作”更稳定地训练：

- 避免 x=0 时幂函数梯度发散
- 避免分母太小
- 避免指数太大导致 overflow

也就是说，他们不是只想着“理论上可导”，而是也处理了“训练时别炸”。

**所以整个反向传播链大概是**  
以最终图像 loss 为例：

text

`final loss -> output sRGB -> gamma -> chroma LUT application (grid_sample) -> LTM/GTM tone curve -> LTM bilateral slice (grid_sample) -> 各子网络参数`

再加上中间 loss：

- cbcr_loss 直接监督 LuTNet
- tm_loss 直接监督 GTM/LTM
- TV loss 直接约束 LUT / LTM map 平滑

所以梯度不仅能“穿过去”，而且作者还故意给了更短的监督路径。

**一句话总结**  
中间虽然有很多“像曲线、像查表”的东西，但它们都被实现成了：

- 连续参数化公式
- 可微分插值采样
- 软 histogram
- 数值稳定版幂函数

所以 PyTorch 能把梯度从最终 loss 一路反传回所有子模块。


