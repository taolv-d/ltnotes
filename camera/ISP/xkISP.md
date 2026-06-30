---
type: note
status: done
tags:
  - camera
  - isp
rating: 0
create: 2026-05-23
update:
---

项目地址：[openasic-org/xkISP: xkISP：Xinkai ISP IP Core (HLS)](https://github.com/openasic-org/xkISP)
xkISP is an open source image signal processor (ISP) based on Xilinx development tools
![[attachments/Pasted image 20260523152538.png]]


以下为GPT总结

## 坏点校正


DPC 使用 5x5 窗口。由于 Bayer 图中相同颜色隔一个像素出现，当前中心点周围同色邻域位于：

```text
(-2,-2), (-2,0), (-2,2),
( 0,-2),         ( 0,2),
( 2,-2), ( 2,0), ( 2,2)
```

设中心像素为 `p0`，8 个同色邻居为 `p_i`。中心点相对所有同色邻居都显著偏高或偏低，认为它是坏点。

```text
p0 - p_i > th_w, for all i  白点

p_i - p0 > th_b, for all i  黑点
```

修正值一般使用 8 个同色邻居的中值：

```text
dst = median(p_1, p_2, ..., p_8)
```

## RAWDNS：RAW 域 NLM

NLM 原理见 [[../denoise/2007 BM3D NLM|2007 BM3D NLM]]

硬件中直接计算指数函数代价高，因此实现通常采用：

- 距离 `D` 使用整数差平方累加。
- `1 / h^2` 预先转成 `invksigma2`。
- 权重通过查表、分段近似或位移实现。
- 最后用整数加权和除以权重和。

```cpp
for each pixel p:
  center_patch = window around p
  sum_w = 0
  sum_v = 0

  for each candidate q in search_window:
    if cfa(q) != cfa(p):
      continue
    dist = patch_distance(p, q)
    w = Cal_weight(dist, reg.invksigma2, reg.Filterpara)
    sum_w += w
    sum_v += w * I(q)

  out = sum_w > 0 ? sum_v / sum_w : I(p)
```


## AWB

这里找白点的算法： Grey World、White Patch、Shades of Grey，以及饱和像素剔除。参考[[AWB/AWB algorithm]]

校正过程

```cpp
uint2 cfa = get_cfa(x, y, imgPattern);
int gain = select_gain(cfa, reg.m_nR, reg.m_nGr, reg.m_nGb, reg.m_nB);
// Gr/Gb 分开设置，因为真实 sensor 中两个绿色通道可能存在响应差异；
int out = (src * gain + 2048) >> 12;
dst.write(clip(out, 0, 4095));
```

## Green Balance

Bayer 有两个绿色采样位置 Gr 和 Gb。理想情况下它们响应相同，但真实 sensor 受像素结构、读出电路和镜头影响，Gr/Gb 会有细微差异。差异会在 demosaic 后变成 maze artifact 或绿色网格纹。

Green Balance 的目标是局部估计 Gr 与 Gb 的偏差，并对中心像素做补偿。

以 7x7 窗口为例，围绕当前像素收集局部 Gr-Gb 差值。为了避免边缘污染，只使用差值绝对值小于阈值的样本：

```text
d_i = Gr_i - Gb_i
if abs(d_i) < threshold:
  accumulate d_i

局部平均差：
d_mean = sum(d_i) / count
```

Gr Gb 按一半差值修正：

```text
Gr' = Gr - d_mean / 2

Gb' = Gb + d_mean / 2
```

```cpp
for each pixel:
  collect local Gr/Gb paired differences in window
  sum = 0
  cnt = 0
  for each pair:
    d = Gr - Gb
    if Lbound <= abs(d) && abs(d) <= Hbound:
      sum += d
      cnt++

  mean = cnt > 0 ? sum / cnt : 0

  if current is Gr-like:
    out = src - mean / 2
  else if current is Gb-like:
    out = src + mean / 2
  else:
    out = src
```

**注意事项**

- 阈值太宽会把边缘颜色差当作 Gr/Gb 偏差，导致边缘变色。
- 阈值太窄会缺少有效样本，修正不稳定。
- GB 放在 WBC 后、Demosaic 前，是为了在颜色恢复前消除绿色通道不一致。

## Demosaic

xkISP 文档描述的是基于梯度和颜色差的边缘自适应算法。看起来是PPG算法[[demosaic/demosaic in dcraw]]

### 1.方向梯度判断

对 R 或 B 位置，先估计水平和垂直方向的变化强度。以 R 点为例（B同理）：

```text
DeltaH_R = |R(m,n-2) + R(m,n+2) - 2R(m,n)| + |G(m,n-1) - G(m,n+1)|

DeltaV_R = |R(m-2,n) + R(m+2,n) - 2R(m,n)| + |G(m-1,n) - G(m+1,n)|
```

直观理解：

- 第一项是同色 R 的二阶差分，用来衡量该方向的亮度弯曲程度。
- 第二项是相邻 G 的一阶差分，用来衡量绿色通道边缘强度。

如果 `DeltaH_R < DeltaV_R`，说明水平方向更平滑，应沿水平插值；反之沿垂直插值。

### 2.方向投票

每个局部点给出水平或垂直判断，统计投票和：

```text
sum >= 4 : 垂直方向占优
sum <= 1 : 水平方向占优
else     : 方向不明确，使用平滑/综合策略
```

投票的意义是降低噪声导致的方向误判。单个像素的梯度可能受噪声、坏点、纹理影响，多点投票更稳定。

### 3.G 通道插值

在 R/B 位置，需要先恢复 G，因为人眼对亮度最敏感，G 通道也承载最多亮度信息。

以 R 点为例，可以构造四个方向候选。用同色 R 的变化修正 G。背后的假设是局部颜色差 `R-G` 比原始通道更平滑。

```text
G_N = G(m-1,n) + 0.5 * (R(m,n) - R(m-2,n))
G_S = G(m+1,n) + 0.5 * (R(m,n) - R(m+2,n))
G_W = G(m,n-1) + 0.5 * (R(m,n) - R(m,n-2))
G_E = G(m,n+1) + 0.5 * (R(m,n) - R(m,n+2))
```

然后计算四个方向的梯度：

```text
nabla_N, nabla_S, nabla_W, nabla_E
```

权重与梯度成反比：

```text
w_k = 1 / (nabla_k + epsilon)
```

根据方向判断选择：

水平插值：

```text
G = (w_W * G_W + w_E * G_E) / (w_W + w_E)
```

垂直插值：

```text
G = (w_N * G_N + w_S * G_S) / (w_N + w_S)
```

方向不明确：

```text
G = (w_N*G_N + w_S*G_S + w_W*G_W + w_E*G_E)
    / (w_N + w_S + w_W + w_E)
```

### 4.R/B 通道插值

恢复 G 后，再恢复缺失的 R/B。常用策略是插值颜色差，颜色差通常比原始通道更平滑，因此这种方法能减少 false color：

```text
D_RG = R - G
D_BG = B - G
```

对于缺失 R 的像素：

```text
R = G + interpolate(D_RG)
```

对于缺失 B 的像素：

```text
B = G + interpolate(D_BG)
```

## EE：边缘增强

中算法是典型 unsharp mask。本质上是增强原图相对低通图的差异。

```text
低频图
L = Gaussian5x5(I)

高频分量：
H = I - L

增强输出：
O = I + k * H    k是锐化强度
```

展开可得：

```text
O = (1 + k)I - kL
```

## CMC：颜色校正矩阵与色彩假彩抑制

### 1.基础矩阵校正

最基本的颜色校正为：

```text
R' = m00*R + m01*G + m02*B + b0
G' = m10*R + m11*G + m12*B + b1
B' = m20*R + m21*G + m22*B + b2
```

定点实现：

```text
R' = (m00*R + m01*G + m02*B + offset + rounding) >> shift
```

### 2.CFC：按 hue 和 edge 调节的假彩抑制

CMC 模块中还包含 CFC 相关参数。根据文档和头文件，可以理解为：

1. 将 RGB 转换成 hue。
2. 判断 hue 是否落在指定范围。
3. 根据边缘强度判断是否位于容易产生假彩的区域。
4. 计算一个抑制比例，调整色彩增益或色差强度。

色相计算可用 HSV 思路：

```text
maxc = max(R, G, B)
minc = min(R, G, B)
delta = maxc - minc
```

若 `maxc == R`：

```text
H = 60 * ((G - B) / delta mod 6)
```

若 `maxc == G`：

```text
H = 60 * ((B - R) / delta + 2)
```

若 `maxc == B`：

```text
H = 60 * ((R - G) / delta + 4)
```

硬件中通常不用浮点角度，而是用比例和查表得到离散 hue。

HueRatio 可按范围和过渡带构造：

```text
inside hue_start..hue_end      -> ratio = 1
outside but within bandshift   -> ratio 从 1 过渡到 0
far outside                    -> ratio = 0
```

EdgeRatio 可按边缘强度构造：

```text
edge <= edge_th                -> ratio = 0
edge_th..edge_th+bandshift     -> ratio 渐增
edge very strong               -> ratio = 1
```

最终抑制强度：

```text
cfc_ratio = HueRatio * EdgeRatio * cfc_strength
```

## GTM：全局色调映射

### 1.LUT 分段插值

`gtmTab[129]` 表示把输入范围分成 128 段。中间点线性插值得到：

```text
out = y0 + (y1 - y0) * frac / 128
```

### 2.Dithering

当高位宽映射到较低位宽时，直接截断会产生 banding。Dithering 通过加入小幅伪随机或误差反馈扰动，使量化误差空间分散。

源码中的逻辑类似：

```text
out = (y_pos0 << 2) + interpolated + Seed
Seed = out & 0x1f
out = out >> 2
```

这里 `Seed` 取低位反馈到下一个像素，可降低平滑渐变中的条带感。


## LTM：局部色调映射
### 1.亮度估计

源码中的 luma 近似为：

```text
L = (84*R + 168*G + 4*B) >> 8 （权重： 0.328 0.656 0.016）
```

### 2.Log 域处理

LTM 对亮度取 log，使用 `log10tab` 查表，避免硬件中计算对数。好处：

- 乘性光照变化变成加性变化。
- 更符合人眼对亮度的感知。

### 3.9x9 bilateral filter

LTM 需要把亮度分成 base layer 和 detail layer：

```text
base   = bilateral_filter(logL)
detail = logL - base
```

### 4.Base/detail 合成

得到 base 和 detail 后，局部 tone mapping 可写为：

```text
out_logL = contrast * base + detail
```

如果 `contrast < 1`，base 动态范围被压缩；detail 保留，使局部纹理不被压平。

然后用指数表恢复线性亮度：

```text
out_L = 10^(out_logL)
```

### 5.RGB 重缩放

为了保持颜色比例，LTM 不直接重新算 RGB，而是用亮度比例缩放原始 RGB：

```text
ratio = out_L / L
R' = R * ratio
G' = G * ratio
B' = B * ratio
```

## CAC：色差校正

CAC 用于修正 chromatic aberration。镜头色差会让 R/B 通道相对 G 通道在边缘处错位，表现为紫边、绿边或彩色轮廓。

CAC 位于 LTM 后、CSC 前，在 RGB 域工作。

G 通道通常空间分辨率最高、亮度贡献最大，CAC 把 G 作为几何边缘参考。它不直接平滑 R/B，而是在边缘附近约束色差：

```text
R-G
B-G
```

如果某个边缘位置的 R-G 或 B-G 突然超出两侧稳定区域的合理范围，就把它 clamp 回合理范围，然后重建 R/B：

```text
R' = G + clamp(R-G)
B' = G + clamp(B-G)
```

CAC 不应替代镜头标定；它是局部修正，不是几何重采样级别的完整色差模型。

### 算法

7x7 RGB 窗口。对 R/G/B 分别计算水平和垂直方向边缘，形式接近 Sobel 或差分核：

```text
edge_h_C = horizontal_gradient(C_window)
edge_v_C = vertical_gradient(C_window)
```

中心 G 边缘超过 `t_transient` 时，认为当前位置可能处于强边缘，需要检查色差。

然后沿水平或垂直方向寻找稳定边界：

```text
left/right 或 up/down 方向上，edge < t_edge 的位置
```

稳定边界处的色差提供合理范围：

```text
low  = min(diff_boundary_1, diff_boundary_2)
high = max(diff_boundary_1, diff_boundary_2)
```

中心或中间区域的色差被限制：

```text
diff' = clamp(diff_center, low, high)
```


## CSC
RGB 到 YUV 色彩空间转换

BT.601：

```text
Y = 0.299R + 0.587G + 0.114B
U = 0.564(B - Y)
V = 0.713(R - Y)
```

BT.709：

```text
Y = 0.2126R + 0.7152G + 0.0722B
U = 0.539(B - Y)
V = 0.635(R - Y)
```

BT.2020：

```text
Y = 0.2627R + 0.6780G + 0.0593B
U = 0.5315(B - Y)
V = 0.678(R - Y)
```

工程中通常写成矩阵：

```text
[Y]   [c00 c01 c02] [R]   [offsetY]
[U] = [c10 c11 c12] [G] + [offsetU]
[V]   [c20 c21 c22] [B]   [offsetV]
```

### 16.3 定点实现

`coeff[12]` 可按 3x4 组织：

```text
[c00 c01 c02 offsetY]
[c10 c11 c12 offsetU]
[c20 c21 c22 offsetV]
```

每个输出通道：

```text
Y = (c00*R + c01*G + c02*B + offsetY + rounding) >> shift
```

U/V 一般需要加中性偏置。例如 10 bit full-range 中：

```text
neutral chroma = 512
```

所以无色灰阶应满足：

```text
U ~= 512
V ~= 512
```

### 16.4 实现骨架

```cpp
int y = (c00*r + c01*g + c02*b + off_y + round) >> shift;
int u = (c10*r + c11*g + c12*b + off_u + round) >> shift;
int v = (c20*r + c21*g + c22*b + off_v + round) >> shift;

dst.write(pack10(clip10(y), clip10(u), clip10(v)));
```

### 16.5 调参与风险

- 系数范围错误会导致 Y 溢出或 U/V 偏色。
- RGB 输入如果仍是线性光，直接转 YUV 与显示标准的 gamma 编码 YUV 不是一回事；需要明确 pipeline 的 transfer function。
- U/V 偏置错误会让灰色带色。

## YFC：YUV 格式转换


YFC 将 YUV444 转为 YUV422 或 YUV420。YUV444 每个像素都有 Y/U/V；YUV422 水平方向两个像素共享一组 U/V；YUV420 水平和垂直方向 2x2 像素共享一组 U/V。

**YUV422**

对相邻两个像素，输出保留两个亮度，色度取平均：

```text
U01 = (U0 + U1 + 1) / 2
V01 = (V0 + V1 + 1) / 2
```

输出布局可为：

```text
Y0 U01 Y1 V01
```

**YUV420**

对 2x2 像素块，亮度全部保留，色度取 4 个像素平均：

```text
U = (U00 + U01 + U10 + U11 + 2) / 4
V = (V00 + V01 + V10 + V11 + 2) / 4
```


## YUVDNS：YUV 域 NLM

YUVDNS 在 CSC/YFC 后处理 YUV 图像。相比 RAWDNS，它工作在更接近视觉和编码的域：

- Y 通道对应亮度噪声。
- U/V 通道对应色度噪声。

色度噪声通常比亮度噪声更容易被人眼接受更强的平滑，因此 Y 和 UV 使用不同参数。

**与 RAWDNS 的差异**

RAWDNS：
- 工作在 CFA 采样结构中。
- 必须注意同色采样位置。
- 主要处理 sensor 原始噪声。

YUVDNS：
- 工作在完整 YUV 图像或子采样 YUV 图像中。
- 可按人眼感知分开处理亮度和色度。
- 适合抑制 demosaic、sharpen、tone mapping 后显露出的残余噪声。

## Scaledown

缩小

```cpp
for out_y:
  for out_x:
    sum_y = sum_u = sum_v = 0
    for dy in 0..times-1:
      for dx in 0..times-1:
        p = input[out_y*times + dy][out_x*times + dx]
        sum_y += p.y
        sum_u += p.u
        sum_v += p.v
    out.y = round(sum_y / (times*times))
    out.u = round(sum_u / (times*times))
    out.v = round(sum_v / (times*times))
```

## Crop
裁剪
```cpp
for y in 0..height-1:
  for x in 0..width-1:
    p = src.read()
    if reg.enable && inside_roi(x, y):
      dst.write(p)
    else if !reg.enable:
      dst.write(p)
```



## 硬件设计与仿真实现

### 为什么中间 FIFO 深度只有 2

`depth=2` 的中间 FIFO 是典型 HLS dataflow 写法。它的意义是：

- 给上下游模块提供最小握手弹性。
- 避免为每一级分配大 FIFO，节省 BRAM/LUTRAM。
- 让图像数据主要以 streaming 方式前进，而不是 frame buffer 方式存取。

这也带来约束：

- 每个模块平均消费/生产速率必须接近。
- 如果某个模块偶发 stall，只有 2 个元素缓冲，很快会反压上游。
- 因此复杂模块内部要通过 pipeline/unroll/partition 保证 II 尽量接近 1。

这套设计的核心思路是：

```text
小 FIFO 串模块，大 line buffer 存局部邻域，整帧不落 DDR。
```

### 行缓存与滑动窗口

ISP 硬件里最重要的面积结构是 line buffer。因为输入是按 raster scan 顺序来的：
如果算法需要 `K x K` 窗口，不能等整帧随机访问，而是保存最近 `K-1` 行，再用寄存器窗口保存当前列附近的 `K` 个像素。

通用结构是：

```text
lineBuffer[K-1][MAX_WIDTH]
window[K][K]

每来一个新像素：
  1. window 每行左移一格
  2. 从 lineBuffer[*][col] 读出前 K-1 行的当前列，填入 window 右侧
  3. 新像素填入 window 最右下角
  4. 把 window 右侧若干值回写 lineBuffer，供未来行使用
  5. 当窗口有效后，处理 window 中心像素
```

该工程里的典型缓存规模如下：

```text
DPC       : rawWindow[5][5], lineBuffer[4][8192]
Demosaic  : rawWindow[5][5], lineBuf[4][8192]
GB        : gb_block[7][7], gb_lines[6][8192]
RAWDNS    : rawdns_block[11][11], rawdns_lines[10][8192]
EE        : ee_block[5][5], ee_lines[4][8192]
LTM       : R/G/B 各 rWindow[9][9], rlineBuf[8][4096]
CAC       : rgbWindow[7][7], lineBuffer[6][8192]
YUVDNS    : Y/U/V 各 yWindow[9][9], ylineBuf[8][8192]
```

### 边界延迟与输出对齐

使用 `K x K` 窗口时，中心像素只有在读入足够行列后才有效。例如 5x5 窗口中心是 `[2][2]`，必须至少读入前 2 行、前 2 列。

代码中常见判断：

```text
if (row > radius && col > radius)
  process center pixel
else
  pass through / output 0 / delay
```

DPC、Demosaic、EE、LTM、CAC、YUVDNS 都有窗口填充和尾部 flush 逻辑。尾部 flush 的作用是把已经进入 line buffer/window 但还没有输出的最后几行/列继续吐出来。

硬件上必须保证：

```text
输入像素数 = frameWidth * frameHeight
输出像素数 = 对应模块期望输出数
```

如果某个模块少吐或多吐一个像素，后续 stream 会永久错位，最终 dataflow deadlock 或输出图像通道错位。

### 定点化策略

工程大量使用 `ap_uint/ap_int` 或 Catapult 的 `ac_int`，不使用浮点。典型位宽包括：

```text
RAW pixel      : uint12
RGB after DMC  : uint36 = 3 * 12
RGB tone stage : uint42 = 3 * 14
YUV            : uint30 = 3 * 10
output plane   : uint10
```

内部乘加会临时扩宽，例如 DGain：

```cpp
int34 dst_tmp;
int22 dst_val;
uint20 gain_w;
uint9  blc_w;
```

计算：

```text
dst_tmp = (src - blc) * gain + 2^11
dst_val = (dst_tmp >> 12) + top_blc
```

这里的硬件意义是：

- 增益 Q12，乘法后需要 `12 + 20 = 32` bit 以上保存结果。
- 加 rounding 常数后右移，避免系统性向下偏差。
- 输出前 clip 到目标位宽，避免 wrap-around。

LSC 中双线性插值也使用预计算倒数：

```text
blockWidth_1
blockHeight_1
```

避免硬件除法。公式形式是：

```text
tmp = ((delta * count * reciprocal + rounding) >> shift)
```

GTM/LTM/YUVDNS/RAWDNS 使用 LUT 或离散权重表替代 `pow/exp/log`：

```text
GTM      : gtmTab[129]
LTM      : log10tab, exp10tab, exptab, pos_kerTab
RAWDNS   : weight_1/weight_2
YUVDNS   : weight calculation with invH2
```

这些都是面积和时序友好的设计。

### 统计类模块如何硬件化

AWB、AEC、AFC 这类模块和逐像素滤波不同，它们的主要输出是统计量或控制量。

硬件统计模块通常有三种结构：

```text
逐像素累加器
  sum += value
  count += valid

分区/直方图统计
  bin[value] += 1

窗口统计
  使用 line buffer/window 或 column statistic 维护局部和
```

本工程中 AWB 当前接在主链路里，但实现形态更接近“边读边统计/透传”。AEC/AFC 源码没有接入主 top，但体现了硬件统计模块的典型方式：

- AEC：逐像素算亮度并累计亮度均值。
- AFC：在指定 ROI 内用 Sobel/Laplace/H5/H9 计算高频能量。
- GB：使用 `ColumnStatistic`、`Gfifo`、`countfifo` 维护局部 Gr/Gb 统计，避免每个像素重新遍历整个窗口。

统计模块的硬件重点不是乘法多复杂，而是：

- 累加器位宽必须足够，否则一帧累加会溢出。
- 统计输出往往应在帧尾更新，下一帧使用。
- 视频系统中常用 shadow register，避免半帧参数突变。

`top_register` 里有 `shadowEb`，说明作者考虑了寄存器影子更新，但当前源码中是否完整实现帧边界切换，需要进一步逐模块核查。

### 面积节省手段

这套代码里能看到几类明确的面积控制策略。

第一，所有算法都定点化：

```text
浮点除法/指数/对数 -> 定点乘法、移位、查表
```

第二，stream 串接避免 frame buffer：

```text
模块间 depth=2 FIFO
局部算法只保存 K-1 行
不把每一级中间图写回 DDR
```

第三，复杂函数限制实例数量。`top_directives.tcl` 中有：

```tcl
set_directive_allocation -type function -limit 3 "ltm" bilaterS
set_directive_allocation -type function -limit 3 "yuv444dns" yuvdns_nlm
```

这表示即使代码上有多个调用，HLS 也最多生成 3 个硬件实例，通过共享降低面积。代价是调度可能变紧，II 或 latency 可能增加。

第四，Catapult 配置以 area 为设计目标：

```tcl
directive set -DESIGN_GOAL area
directive set -REGISTER_SHARING_MAX_WIDTH_DIFFERENCE 8
directive set -MERGEABLE true
directive set -OPT_CONST_MULTS use_library
```

这说明 Catapult 路径明确偏向面积优化，而不是无节制展开。

第五，大表和配置项映射成 direct input 或 register：

```text
CMC gain
GTM table
CSC coeff
LSC gain table
```

direct input/register 的好处是读取延迟固定，不需要复杂总线访问；坏处是表很大时端口数量和扇出会成为时序问题。因此 Vivado 指令里对表做 array partition，Catapult 里对 `gtmTab` 还有 packing 设置。

### 吞吐优化手段

吞吐优化主要体现在：

```tcl
set_directive_pipeline "<module>/<row_or_col_loop>"
set_directive_unroll "<module>/<small_loop>"
set_directive_array_partition ...
set_directive_dependence -dependent false ...
```

逐像素 row/col loop 被 pipeline，目标是一个像素一个周期。小循环如窗口 shift、矩阵乘加、Gaussian taps、edge taps 被 unroll，使组合逻辑并行完成。

`set_directive_dependence -dependent false` 用于告诉 HLS 某些数组访问没有真实跨迭代依赖。例如 GB 中：

```tcl
set_directive_dependence -variable Gfifo -type inter -dependent false "gb_process"
set_directive_dependence -variable countfifo -type inter -dependent false "gb_process"
```

这是 HLS 优化里很重要的一点。HLS 工具保守时会认为数组在相邻循环迭代间有读写相关，从而不能 II=1。作者显式解除依赖，换取 pipeline。

但这类 directive 也有风险：如果代码确实存在跨迭代真实依赖，强行声明 false 会导致综合出的硬件行为和 C 仿真不一致。因此它需要和访问模式严格匹配。


### 按模块看硬件复杂度

从硬件面积和吞吐角度，可以把模块分成几类。

低成本逐像素模块：

```text
TPG, DGain, AWB透传部分, WBC, CMC, GTM, CSC, YFC, Crop
```

特点：

- 不需要大窗口。
- 主要是乘加、查表、clip。
- line buffer 需求小或没有。
- 容易做到 II=1。

中等成本窗口模块：

```text
DPC, Demosaic, EE, GB, Scaledown
```

特点：

- 需要 5x5 或 7x7 窗口。
- line buffer 使用 4 到 6 行。
- 小窗口 complete partition，逻辑并行度较高。
- 面积主要来自窗口逻辑、比较器、加法树和 BRAM。

高成本复杂窗口模块：

```text
RAWDNS, LTM, CAC, YUVDNS
```

特点：

- 需要 7x7、9x9、11x11 级别邻域。
- 计算包括权重、距离、双边/NLM、边缘检测或多通道处理。
- line buffer 多，窗口大，加法树和乘法多。
- directives 中出现 function allocation limit，说明作者有意识控制面积。

对面积优化最敏感的模块通常是：

```text
RAWDNS / YUVDNS / LTM / CAC
```