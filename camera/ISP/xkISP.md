项目地址：[openasic-org/xkISP: xkISP：Xinkai ISP IP Core (HLS)](https://github.com/openasic-org/xkISP)
xkISP is an open source image signal processor (ISP) based on Xilinx development tools
![[attachments/Pasted image 20260523152538.png]]


以下为GPT总结 [[../../TODO|TODO]]

# xkISP 主模块算法深度解析

本文面向 `isp/xkISP-main/xkISP-main` 工程，按 `src/top.cpp` 中实际调用顺序解释 xkISP 主链路中每个模块的算法、数学原理、定点实现方式、寄存器参数含义和工程实现要点。目标不是复述模块说明，而是把每个模块在 ISP pipeline 中“为什么做、怎么做、代码大致如何落地、调参会影响什么”讲清楚。

## 0. 资料范围与主线

参考文件主要包括：

- `src/top.cpp`：主 pipeline 调用顺序。
- `src/top.h`：顶层寄存器、像素位宽、HLS stream 类型。
- `src/*.{h,cpp}`：各模块寄存器定义与部分实现。
- `doc/CHN/*`：各模块算法说明文档。

`top.cpp` 的主链路可以概括为：

```text
RAW12 input
  -> TPG
  -> DGain
  -> LSC
  -> DPC
  -> RAWDNS
  -> AWB
  -> WBC
  -> Green Balance
  -> Demosaic
  -> Edge Enhancement
  -> CMC
  -> GTM
  -> LTM
  -> CAC
  -> CSC
  -> YFC
  -> YUVDNS
  -> Scaledown
  -> Crop
  -> store_out
```

这是一条典型从 Bayer RAW 到 YUV 输出的 ISP 流水线。前半段保留 Bayer 采样结构，主要处理传感器和镜头带来的非理想因素；中段完成颜色恢复和色彩校正；后半段进入 RGB/YUV 图像域，处理 tone mapping、锐化、色差、降噪、缩放和裁剪。

## 1. 顶层数据流与定点格式

### 1.1 像素流类型

`top.h` 中使用 Xilinx HLS 风格的数据类型和 `hls::stream` 串接模块。主要像素位宽如下：

```text
stream_u12 : RAW 或单通道像素，uint12
stream_u36 : RGB，每通道 12 bit，共 36 bit
stream_u42 : RGB，每通道 14 bit，共 42 bit
stream_u30 : YUV，每通道 10 bit，共 30 bit
stream_u10 : 输出 Y/U/V 单通道 10 bit
```

这说明系统在 RAW 域使用 12 bit，demosaic 后先得到 12 bit RGB，后续 CMC/GTM/LTM/CAC 使用 14 bit RGB 保存更大的中间动态范围，CSC 后压到 10 bit YUV。

这种位宽安排的原因是：

- RAW 输入通常来自 10/12 bit sensor，这里以 12 bit 为基准。
- 白平衡、LSC、CMC、tone mapping 都可能放大信号，需要额外 headroom。
- 最终 YUV 输出常用于视频或图像编码，10 bit 是常见工程格式。

### 1.2 顶层寄存器

`top_register` 中包含整条链路共享的基础信息：

```text
frameWidth / frameHeight : 当前帧尺寸
inputFormat              : 输入格式
imgPattern               : Bayer pattern
pipeMode                 : pipeline 模式
blc                      : 全局 black level
shadowEb                 : shadow 寄存器开关
binningFrameWidth/Height : binning 后尺寸
scalerFrameWidth/Height  : scaler 后尺寸
```

其中最关键的是 `imgPattern` 和 `blc`。

`imgPattern` 用于确定当前坐标 `(x, y)` 是 R、Gr、Gb 还是 B。源码中常见计算形式为：

```text
bayerPattern = (((y & 1) << 1) + (x & 1)) ^ imgPattern
```

坐标低位给出 2x2 Bayer 单元中的相对位置，和 `imgPattern` 异或后得到实际颜色通道。这样同一套处理逻辑可兼容 RGGB、GRBG、GBRG、BGGR 等排列。

`blc` 是全局黑电平。很多模块在计算前会先把传感器黑电平扣除，增益后再加回统一黑位：

```text
linear_value = input - channel_black_level
out = gain(linear_value) + top_reg.blc
```

这能避免把 offset 当作真实光信号参与增益放大。

### 1.3 HLS 流式处理模型

大部分模块都采用逐像素流式处理：

```text
for y in [0, height):
  for x in [0, width):
    pixel = src.read()
    out   = process(pixel, x, y, local_window, registers)
    dst.write(out)
```

涉及邻域的模块会用行缓存和滑动窗口，例如 DPC 的 5x5、RAWDNS 的较大窗口、Demosaic 的 5x5 或更大邻域、CAC 的 7x7、LTM 的 9x9。HLS 中通常不会随机访问整张图，而是通过 line buffer 保存最近若干行，并用 shift register 形成当前窗口。

## 2. TPG：测试图案生成

### 2.1 模块位置与目的

TPG 是 pipeline 第一个模块。它的作用是产生可控测试图案，便于验证后续 ISP 模块，不依赖真实 sensor 输入。

如果 `tpg_register.eb == 0`，通常应直接透传输入 RAW；如果开启，则生成指定 Bayer pattern 下的彩条。

### 2.2 关键参数

`tpg_register` 主要包含：

```text
eb          : 使能
width       : 图案宽度
height      : 图案高度
imgPattern  : Bayer pattern
rolling     : 是否滚动
sensor_timing_* : 模拟传感器 timing
id          : pattern id
```

### 2.3 算法原理

源码里的测试图案按宽度分成 8 个 block，每个 block 对应一种颜色：

```text
white, black, red, green, blue, cyan, magenta, yellow
```

每个像素先根据 `(x, y, imgPattern)` 判断当前 Bayer 位置对应的颜色通道，然后根据所在色块决定该通道输出 0 还是满量程。

12 bit RAW 满量程为：

```text
MAXS = 4095
```

例如在红色块中：

- R 采样点输出 4095。
- Gr/Gb/B 采样点输出 0。

在白色块中：

- R/Gr/Gb/B 都输出 4095。

在黄色块中：

- R 和 G 采样点输出 4095。
- B 采样点输出 0。

### 2.4 实现骨架

```cpp
for (int y = 0; y < height; ++y) {
  for (int x = 0; x < width; ++x) {
    uint2 cfa = (((y & 1) << 1) + (x & 1)) ^ imgPattern;
    int block = x * 8 / width;
    uint12 out = ColorSelect(cfa, block);
    dst.write(out);
  }
}
```

### 2.5 工程注意事项

- TPG 是验证 ISP 链路的基础。彩条可以快速暴露 Bayer pattern 配置错误、通道交换、白平衡增益异常和色彩矩阵错误。
- 如果启用 rolling，图案随帧或行移动，可用于验证帧同步和缓存边界。
- 宽度不能太小，否则 8 个 block 的分割会退化。

## 3. DGain：数字增益与黑电平处理

### 3.1 模块位置与目的

DGain 位于 RAW 初始阶段，处理 sensor 输出的黑电平和数字增益。它是 RAW 域最基础的线性校正模块。

目的包括：

- 按通道扣除不同 black level。
- 按 R/Gr/Gb/B 分别施加 digital gain。
- 把信号重新对齐到统一黑电平 `top_reg.blc`。

### 3.2 关键参数

`dgain_register` 中包含：

```text
eb       : 使能
m_nBlcR  : R 黑电平
m_nBlcGr : Gr 黑电平
m_nBlcGb : Gb 黑电平
m_nBlcB  : B 黑电平
m_nR     : R 增益
m_nGr    : Gr 增益
m_nGb    : Gb 增益
m_nB     : B 增益
```

增益使用定点数，源码里有：

```text
GAIN_BITS = 12
GAIN_HALF_VALUE = 1 << 11
```

所以增益通常可理解为 Q*.12 格式，`4096` 表示 1.0x。

### 3.3 数学原理

对每个 RAW 像素，先根据 Bayer 位置选择通道参数：

```text
blc_c  = BlcR / BlcGr / BlcGb / BlcB
gain_c = GainR / GainGr / GainGb / GainB
```

然后执行：

```text
v0 = src - blc_c
v1 = v0 * gain_c
v2 = round(v1 / 2^12)
dst = clip(v2 + top_blc, 0, 4095)
```

源码中的定点舍入形式是：

```text
dst_tmp = (src - blc_c) * gain_c + 2^(GAIN_BITS - 1)
dst_val = (dst_tmp >> GAIN_BITS) + top_reg.blc
```

这等价于四舍五入，而不是单纯截断。

### 3.4 实现骨架

```cpp
uint2 cfa = (((y & 1) << 1) + (x & 1)) ^ top_reg.imgPattern;

switch (cfa) {
  case R:  blc = reg.m_nBlcR;  gain = reg.m_nR;  break;
  case Gr: blc = reg.m_nBlcGr; gain = reg.m_nGr; break;
  case Gb: blc = reg.m_nBlcGb; gain = reg.m_nGb; break;
  case B:  blc = reg.m_nBlcB;  gain = reg.m_nB;  break;
}

int tmp = (int(src) - int(blc)) * int(gain) + (1 << 11);
int out = (tmp >> 12) + top_reg.blc;
dst.write(clip(out, 0, 4095));
```

### 3.5 调参与风险

- 黑电平过小：暗部偏灰，后续白平衡和 tone mapping 会放大暗部噪声。
- 黑电平过大：暗部被截断，阴影细节消失。
- RAW 域增益过大：提升亮度，但同时放大 shot noise、read noise 和坏点。
- R/Gr/Gb/B 增益不一致时，可能和 WBC/AWB 功能重叠，需要明确哪个模块负责最终白平衡。

## 4. LSC：镜头阴影校正

### 4.1 模块位置与目的

LSC 处理 lens shading，即镜头和 sensor 微透镜导致的空间非均匀性。典型现象是画面中心亮、四角暗，同时不同颜色通道的衰减程度不同。

LSC 在 RAW 域执行，因为镜头阴影是 CFA 采样前的光学/传感器效应，按 R/Gr/Gb/B 分开校正更合理。

### 4.2 关键参数

`lsc_register` 包含：

```text
eb                : 使能
rGain[13*17]      : R 网格增益
GrGain[13*17]     : Gr 网格增益
GbGain[13*17]     : Gb 网格增益
bGain[13*17]      : B 网格增益
meshScale         : mesh 缩放参数
blockWidth        : 水平方向 block 宽度
blockHeight       : 垂直方向 block 高度
blockWidth_1      : blockWidth 的倒数定点近似
blockHeight_1     : blockHeight 的倒数定点近似
```

文档描述为 17x13 个 mesh 顶点，对应 16x12 个图像块。每个 CFA 通道都有一张增益表。

### 4.3 数学原理

图像坐标 `(x, y)` 落在某个 mesh cell 中：

```text
cell_x = floor(x / blockWidth)
cell_y = floor(y / blockHeight)
```

cell 四个角的增益为：

```text
g00 = G[cell_y    ][cell_x    ]
g10 = G[cell_y    ][cell_x + 1]
g01 = G[cell_y + 1][cell_x    ]
g11 = G[cell_y + 1][cell_x + 1]
```

像素在 cell 内的归一化位置：

```text
u = (x - cell_x * blockWidth) / blockWidth
v = (y - cell_y * blockHeight) / blockHeight
```

双线性插值得到当前位置增益：

```text
g(x,y) =
  (1-u)(1-v)g00
  + u(1-v)g10
  + (1-u)v g01
  + uv g11
```

输出为：

```text
dst(x,y) = clip(src(x,y) * g(x,y), 0, 4095)
```

实际硬件实现会把除法改成乘倒数或移位，用 `blockWidth_1`、`blockHeight_1` 做定点近似。

### 4.4 实现骨架

```cpp
uint2 cfa = get_cfa(x, y, imgPattern);
const uint16_t* table = select_lsc_table(cfa);

int bx = min(x / blockWidth,  15);
int by = min(y / blockHeight, 11);

int dx = x - bx * blockWidth;
int dy = y - by * blockHeight;

gain = bilinear(table[by][bx], table[by][bx+1],
                table[by+1][bx], table[by+1][bx+1],
                dx, dy);

dst = clip((src * gain + rounding) >> gain_shift);
```

### 4.5 调参与风险

- LSC 表通常来自标定，而不是手工调参。
- 四角增益太大时会明显放大暗角噪声。
- R/B 表不匹配会产生色彩阴影，例如角落偏红或偏蓝。
- LSC 应在 AWB 前执行，否则 AWB 统计会被空间色偏污染。

## 5. DPC：坏点校正

### 5.1 模块位置与目的

DPC 用于修正 sensor 上的 dead pixel、hot pixel 或异常像素。它在 RAW 域处理，使用同色邻域判断当前 CFA 采样点是否异常。

### 5.2 关键参数

`dpc_register` 包含：

```text
eb   : 使能
th_w : 白坏点阈值
th_b : 黑坏点阈值
```

### 5.3 数学原理

DPC 使用 5x5 窗口。由于 Bayer 图中相同颜色隔一个像素出现，当前中心点周围同色邻域位于：

```text
(-2,-2), (-2,0), (-2,2),
( 0,-2),         ( 0,2),
( 2,-2), ( 2,0), ( 2,2)
```

设中心像素为 `p0`，8 个同色邻居为 `p_i`。

白坏点判定：

```text
p0 - p_i > th_w, for all i
```

黑坏点判定：

```text
p_i - p0 > th_b, for all i
```

也就是中心点相对所有同色邻居都显著偏高或偏低，才认为它是坏点。这样能避免把真实边缘误判为坏点。

修正值一般使用 8 个同色邻居的中值：

```text
dst = median(p_1, p_2, ..., p_8)
```

如果不满足坏点条件，则：

```text
dst = p0
```

### 5.4 实现骨架

```cpp
int diff_pos_count = 0;
int diff_neg_count = 0;

for each same_color_neighbor n:
  if (center - n > th_w) diff_pos_count++;
  if (n - center > th_b) diff_neg_count++;

bool hot  = diff_pos_count == 8;
bool dead = diff_neg_count == 8;

if (hot || dead)
  out = median8(neighbors);
else
  out = center;
```

### 5.5 工程注意事项

- 阈值太小：边缘和细纹理会被误修正，导致软化或局部伪影。
- 阈值太大：坏点漏检，后续 demosaic 会把单点异常扩散成彩色斑点。
- DPC 通常应在 RAWDNS 之前，因为坏点属于强异常值，先修正可降低降噪误判。

## 6. RAWDNS：RAW 域非局部均值降噪

### 6.1 模块位置与目的

RAWDNS 在 demosaic 前降噪，目标是抑制 RAW sensor 噪声，同时尽量保留边缘和纹理。它使用 NLM 思路，即不是简单平均邻域，而是按局部块相似度加权平均。

### 6.2 关键参数

`rawdns_register` 包含：

```text
eb          : 使能
sigma       : 噪声强度估计
Filterpara  : 滤波强度参数
invksigma2  : 1 / (k * sigma^2) 的定点形式
```

源码中还有若干辅助函数：

```text
rawdns_abs
rawdns_max
rawdns_clip
Cal_weight
Cal_Eur_Distance
rawdns_process
```

### 6.3 NLM 数学原理

经典 NLM 对像素 `p` 的估计为：

```text
I'(p) = sum_{q in S(p)} w(p,q) I(q)
```

其中 `S(p)` 是搜索窗口，权重由以 `p` 和 `q` 为中心的 patch 相似度决定：

```text
w(p,q) = exp(-D(p,q) / h^2) / Z(p)
```

```text
D(p,q) = sum_{r in patch} (I(p+r) - I(q+r))^2
```

归一化因子：

```text
Z(p) = sum_{q in S(p)} exp(-D(p,q) / h^2)
```

RAW Bayer 中不能随便混合不同颜色采样点，因此实际窗口会考虑 CFA 周期。同色采样点通常间隔 2 像素，比较 patch 时需要保持颜色位置一致。

### 6.4 定点实现

硬件中直接计算指数函数代价高，因此实现通常采用：

- 距离 `D` 使用整数差平方累加。
- `1 / h^2` 预先转成 `invksigma2`。
- 权重通过查表、分段近似或位移实现。
- 最后用整数加权和除以权重和。

近似流程为：

```text
dist = sum(abs_or_square_difference)
weight = Cal_weight(dist, invksigma2, Filterpara)
sum_w += weight
sum_p += weight * candidate_pixel
out = sum_p / sum_w
```

如果 `sum_w == 0`，通常应回退为中心像素。

### 6.5 实现骨架

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

### 6.6 调参与风险

- `sigma` 越大，认为噪声越强，降噪越重。
- `Filterpara` 越强，平坦区域更干净，但纹理更容易被抹掉。
- RAW 域 NLM 应避免跨颜色通道比较，否则会引入彩色伪影。
- 过强 RAWDNS 会让 demosaic 缺少真实高频信息，导致后续锐化产生假边。

## 7. AWB：自动白平衡统计与增益估计

### 7.1 模块位置与目的

AWB 位于 WBC 之前，用于估计当前光源下 R/G/B 通道应当施加的白平衡增益。AWB 本身主要是统计和增益计算，WBC 负责把增益应用到像素。

### 7.2 关键参数

`awb_register` 包含：

```text
eb      : 使能
r_gain  : R 通道输出增益
g_gain  : G 通道输出增益
b_gain  : B 通道输出增益
coeff   : 算法内部系数
```

文档中提到的算法族包括 Grey World、White Patch、Shades of Grey，以及饱和像素剔除。

### 7.3 Grey World 原理

Grey World 假设一幅自然图像的平均反射颜色应为灰色，即：

```text
mean_R ~= mean_G ~= mean_B
```

统计每个颜色通道平均值：

```text
R_avg = sum(R_i) / N_R
G_avg = sum(G_i) / N_G
B_avg = sum(B_i) / N_B
```

以 G 为参考，得到：

```text
gain_R = G_avg / R_avg
gain_G = 1
gain_B = G_avg / B_avg
```

如果使用整体灰度均值 `K = (R_avg + G_avg + B_avg) / 3`，也可写成：

```text
gain_R = K / R_avg
gain_G = K / G_avg
gain_B = K / B_avg
```

### 7.4 White Patch 原理

White Patch 假设画面中最亮区域接近白色。统计每个通道高亮代表值：

```text
R_max, G_max, B_max
```

或者取最亮前 `w_T%` 像素的均值，避免单个噪声点影响。增益为：

```text
gain_R = K / R_white
gain_G = K / G_white
gain_B = K / B_white
```

其中 `K` 可取三个通道 white 值的均值或 G 通道值。

### 7.5 Shades of Grey 原理

Shades of Grey 是 Grey World 和 White Patch 的统一形式，使用 Minkowski 范数：

```text
L_p(C) = (1/N * sum_i C_i^p)^(1/p)
```

当 `p=1` 时接近 Grey World；当 `p` 趋近无穷大时接近 White Patch。

增益：

```text
gain_C = K / L_p(C)
```

### 7.6 饱和像素剔除

饱和像素不再反映真实通道比例，需要从统计中剔除。常见规则：

```text
if R > T or G > T or B > T:
  ignore pixel
```

或只剔除接近满量程的通道。AWB 文档中有 `remove_en`、`remove_T` 之类概念，其目的就是降低高光 clipping 对光源估计的污染。

### 7.7 工程注意事项

- AWB 统计最好在 LSC 后进行，否则暗角和色彩阴影会污染平均值。
- 大面积单色场景会违反 Grey World 假设，例如草地、蓝天、红墙。
- 高亮区域不一定是白色，White Patch 在舞台灯、彩灯场景容易失效。
- AWB 输出增益应做限幅和平滑，否则视频中会出现色温跳变。

## 8. WBC：白平衡校正

### 8.1 模块位置与目的

WBC 根据 AWB 或手动配置的 R/Gr/Gb/B 增益，在 RAW Bayer 域逐像素调整通道强度。它是白平衡真正作用到像素数据的模块。

### 8.2 关键参数

`wbc_register` 包含：

```text
eb     : 使能
m_nR   : R 增益
m_nGr  : Gr 增益
m_nGb  : Gb 增益
m_nB   : B 增益
```

增益为 15 bit，通常按 Q3.12 理解：

```text
4096 = 1.0x
8192 = 2.0x
```

### 8.3 数学原理

对每个 RAW 像素：

```text
dst = clip(src * gain_c / 2^12, 0, 4095)
```

其中 `gain_c` 由 Bayer 位置选择。

Gr/Gb 分开设置很重要，因为真实 sensor 中两个绿色通道可能存在响应差异；同时 GB 模块也会进一步处理 Gr/Gb 不一致问题。

### 8.4 实现骨架

```cpp
uint2 cfa = get_cfa(x, y, imgPattern);
int gain = select_gain(cfa, reg.m_nR, reg.m_nGr, reg.m_nGb, reg.m_nB);
int out = (src * gain + 2048) >> 12;
dst.write(clip(out, 0, 4095));
```

### 8.5 调参与风险

- R/B 增益一般大于 G，低色温时 B 增益较大，高色温时 R 增益较大。
- 增益过大可能导致单通道先 clipping，造成高光偏色。
- WBC 后的噪声也会按通道放大，尤其是低照下的蓝通道。

## 9. Green Balance：双绿通道平衡

### 9.1 模块位置与目的

Bayer 有两个绿色采样位置 Gr 和 Gb。理想情况下它们响应相同，但真实 sensor 受像素结构、读出电路和镜头影响，Gr/Gb 会有细微差异。差异会在 demosaic 后变成 maze artifact 或绿色网格纹。

Green Balance 的目标是局部估计 Gr 与 Gb 的偏差，并对中心像素做补偿。

### 9.2 关键参数

`gb_register` 包含：

```text
eb          : 使能
win_size    : 窗口大小
Lbound      : 低阈值
Hbound      : 高阈值
threhold    : 差异阈值
```

源码中还有 `ColumnStatistic`，说明该模块通过列统计和滑动窗口实现流式局部均值。

### 9.3 算法原理

以 7x7 窗口为例，围绕当前像素收集局部 Gr-Gb 差值。为了避免边缘污染，只使用差值绝对值小于阈值的样本：

```text
d_i = Gr_i - Gb_i
if abs(d_i) < threshold:
  accumulate d_i
```

局部平均差：

```text
d_mean = sum(d_i) / count
```

如果当前点位于 R/Gr 相关位置，可按一半差值修正：

```text
Gr' = Gr - d_mean / 2
```

如果当前点位于 B/Gb 相关位置：

```text
Gb' = Gb + d_mean / 2
```

实际实现会根据当前 Bayer 位置选择是估计 `Gr-Gb` 还是 `Gb-Gr`，并用上下界限制修正幅度。

### 9.4 实现骨架

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

### 9.5 工程注意事项

- 阈值太宽会把边缘颜色差当作 Gr/Gb 偏差，导致边缘变色。
- 阈值太窄会缺少有效样本，修正不稳定。
- GB 放在 WBC 后、Demosaic 前，是为了在颜色恢复前消除绿色通道不一致。

## 10. Demosaic：Bayer 去马赛克

### 10.1 模块位置与目的

Demosaic 将单通道 Bayer RAW 恢复为每个像素都有 R/G/B 三通道的 RGB 图像。它是 ISP 中最关键的图像重建模块之一，因为 RAW 域每个位置只采样一种颜色，另外两种颜色必须由邻域估计。

`demosaic_register` 只有 `eb`，说明该实现主要是固定算法而非高度参数化算法。

### 10.2 基础问题

Bayer pattern 每个 2x2 单元包含：

```text
R  Gr
Gb B
```

或它的其他排列。Demosaic 的目标是对每个坐标 `(m,n)` 估计：

```text
R(m,n), G(m,n), B(m,n)
```

已采样通道直接来自 RAW，未采样通道通过插值估计。

简单双线性插值会在边缘产生 zipper artifact 和 false color。xkISP 文档描述的是基于梯度和颜色差的边缘自适应算法。

### 10.3 方向梯度判断

对 R 或 B 位置，先估计水平和垂直方向的变化强度。以 R 点为例：

```text
DeltaH_R =
  |R(m,n-2) + R(m,n+2) - 2R(m,n)|
  + |G(m,n-1) - G(m,n+1)|

DeltaV_R =
  |R(m-2,n) + R(m+2,n) - 2R(m,n)|
  + |G(m-1,n) - G(m+1,n)|
```

直观理解：

- 第一项是同色 R 的二阶差分，用来衡量该方向的亮度弯曲程度。
- 第二项是相邻 G 的一阶差分，用来衡量绿色通道边缘强度。

如果 `DeltaH_R < DeltaV_R`，说明水平方向更平滑，应沿水平插值；反之沿垂直插值。

B 点同理：

```text
DeltaH_B =
  |B(m,n-2) + B(m,n+2) - 2B(m,n)|
  + |G(m,n-1) - G(m,n+1)|

DeltaV_B =
  |B(m-2,n) + B(m+2,n) - 2B(m,n)|
  + |G(m-1,n) - G(m+1,n)|
```

### 10.4 方向投票

文档中不是只用单点梯度，而是在邻域内做方向投票。每个局部点给出水平或垂直判断，统计投票和：

```text
sum >= 4 : 垂直方向占优
sum <= 1 : 水平方向占优
else     : 方向不明确，使用平滑/综合策略
```

投票的意义是降低噪声导致的方向误判。单个像素的梯度可能受噪声、坏点、纹理影响，多点投票更稳定。

### 10.5 G 通道插值

在 R/B 位置，需要先恢复 G，因为人眼对亮度最敏感，G 通道也承载最多亮度信息。

以 R 点为例，可以构造四个方向候选：

```text
G_N = G(m-1,n) + 0.5 * (R(m,n) - R(m-2,n))
G_S = G(m+1,n) + 0.5 * (R(m,n) - R(m+2,n))
G_W = G(m,n-1) + 0.5 * (R(m,n) - R(m,n-2))
G_E = G(m,n+1) + 0.5 * (R(m,n) - R(m,n+2))
```

这不是单纯平均 G 邻居，而是用同色 R 的变化修正 G。背后的假设是局部颜色差 `R-G` 比原始通道更平滑。

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

### 10.6 R/B 通道插值

恢复 G 后，再恢复缺失的 R/B。常用策略是插值颜色差，而不是直接插值颜色值：

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

颜色差通常比原始通道更平滑，因此这种方法能减少 false color。

在 G 位置估计 R/B 时，会根据该 G 是 Gr 还是 Gb 选择水平或垂直方向的同色邻居。例如在 Gr 位置，R 可能左右相邻，B 可能上下相邻；在 Gb 位置则相反。

### 10.7 输出格式

Demosaic 输出为 `stream_u36`：

```text
R12 | G12 | B12
```

每个通道 12 bit。后续 EE/CMC 会把 RGB 作为完整彩色图像处理。

### 10.8 调参与风险

- 方向判断错误会在边缘产生拉链状 artifact。
- RAWDNS 太强会削弱梯度，造成方向判断偏平滑。
- DPC 漏检的坏点会在 demosaic 中扩散为彩点。
- Gr/Gb 不平衡会让绿色插值出现棋盘纹，所以 GB 应在 demosaic 前完成。

## 11. EE：边缘增强

### 11.1 模块位置与目的

EE 在 demosaic 后处理 RGB 图像，用于补偿光学模糊、demosaic 平滑和降噪导致的锐度下降。文档中算法是典型 unsharp mask。

`ee_register` 包含：

```text
eb    : 使能
coeff : 锐化系数
```

### 11.2 数学原理

先用 5x5 Gaussian blur 得到低频图：

```text
L = Gaussian5x5(I)
```

高频分量：

```text
H = I - L
```

增强输出：

```text
O = I + k * H
```

其中 `k = coeff` 对应锐化强度。

展开可得：

```text
O = (1 + k)I - kL
```

这说明锐化本质上是增强原图相对低通图的差异。

### 11.3 5x5 Gaussian

常见 5x5 高斯核形如：

```text
1  4  6  4 1
4 16 24 16 4
6 24 36 24 6
4 16 24 16 4
1  4  6  4 1
```

归一化因子为 256。即：

```text
L(x,y) = sum(kernel(i,j) * I(x+i,y+j)) / 256
```

实际实现可能使用移位 `>> 8`。

### 11.4 实现骨架

```cpp
for each RGB pixel:
  blur_r = gaussian5x5(R_window)
  blur_g = gaussian5x5(G_window)
  blur_b = gaussian5x5(B_window)

  hf_r = R - blur_r
  hf_g = G - blur_g
  hf_b = B - blur_b

  out_r = clip(R + coeff * hf_r)
  out_g = clip(G + coeff * hf_g)
  out_b = clip(B + coeff * hf_b)
```

### 11.5 调参与风险

- 锐化会放大噪声，尤其是 demosaic 后的彩噪。
- `coeff` 过大会造成 halo 和 overshoot。
- EE 放在 CMC 前意味着锐化在 RGB 线性/近线性域进行，色彩矩阵会进一步影响锐化后的通道。

## 12. CMC：颜色校正矩阵与色彩假彩抑制

### 12.1 模块位置与目的

CMC 用 3x3 或 3x4 矩阵把 sensor RGB 转换到目标 RGB 色彩空间。因为 sensor 的 R/G/B 光谱响应不等于标准 RGB，需要通过矩阵校正色彩。

`cmc_register` 包含：

```text
eb             : CMC 使能
m_nGain[12]    : 矩阵/偏置系数，4096 表示 1X
cf_enable      : CFC 使能
hue_start/end  : 色相范围
edge_th         : 边缘阈值
hue_bandsift   : hue 过渡带宽
edge_bandsift  : edge 过渡带宽
cfc_strength   : 色彩假彩抑制强度
```

`CMC_BITS_DEEP = 14`，说明输出进入 14 bit RGB 域。`CMC_SHIFT_DEEP = 10` 表示某些矩阵计算使用 10 bit 小数或移位。

### 12.2 基础矩阵校正

最基本的颜色校正为：

```text
R' = m00*R + m01*G + m02*B + b0
G' = m10*R + m11*G + m12*B + b1
B' = m20*R + m21*G + m22*B + b2
```

如果 `m_nGain[12]` 存 12 个参数，常见组织方式是 3 行 x 4 列：

```text
[m00 m01 m02 b0]
[m10 m11 m12 b1]
[m20 m21 m22 b2]
```

定点实现：

```text
R' = (m00*R + m01*G + m02*B + offset + rounding) >> shift
```

其中 `4096=1X` 表明系数可按 Q12 理解；源码常量又有 `CMC_SHIFT_DEEP=10`，所以实际移位要以源码实现为准。工程理解上，它就是“整数乘加 + 移位 + clip”。

### 12.3 为什么 CMC 后是 14 bit

矩阵可能产生负值或超过 12 bit 的正值。例如提升红色饱和度时：

```text
R' = 1.5R - 0.2G - 0.1B
```

中间值需要符号位和额外动态范围。输出到 14 bit 可以减少 tone mapping 前的截断。

### 12.4 CFC：按 hue 和 edge 调节的假彩抑制

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

### 12.5 实现骨架

```cpp
RGB14 cmc_apply(RGB12 in) {
  int r = (m00*in.r + m01*in.g + m02*in.b + b0 + round) >> shift;
  int g = (m10*in.r + m11*in.g + m12*in.b + b1 + round) >> shift;
  int b = (m20*in.r + m21*in.g + m22*in.b + b2 + round) >> shift;

  if (reg.cf_enable) {
    hue = RGB2H(r, g, b);
    hue_ratio = calc_hue_ratio(hue);
    edge_ratio = calc_edge_ratio(local_edge);
    ratio = hue_ratio * edge_ratio * reg.cfc_strength;
    suppress_false_color(r, g, b, ratio);
  }

  return clip14(r, g, b);
}
```

### 12.6 调参与风险

- CCM 标定不准会导致整体色偏或饱和度异常。
- 负系数会放大噪声，尤其在暗部。
- CFC 太强会降低彩色边缘的真实饱和度。
- CMC 前后的 black level 和动态范围必须一致，否则矩阵会把 offset 当作颜色参与计算。

## 13. GTM：全局色调映射

### 13.1 模块位置与目的

GTM 是 Global Tone Mapping，用全局 LUT 把输入亮度或 RGB 强度映射到目标动态范围。它处理全局对比度和 gamma 曲线。

`gtm_register` 包含：

```text
eb          : 使能
dithering_en : dithering 使能
gtmTab[129] : tone mapping 查找表
```

### 13.2 LUT 分段插值

`gtmTab[129]` 表示把输入范围分成 128 段。若输入是 14 bit：

```text
input range = 0..16383
segment size = 128
index = input >> 7
frac = input & 0x7f
```

相邻两个 LUT 点为：

```text
y0 = gtmTab[index]
y1 = gtmTab[index + 1]
```

线性插值：

```text
out = y0 + (y1 - y0) * frac / 128
```

源码里可见近似形式：

```text
y_pos0 = (gtmTab[index] << 4) | 0xf
y_pos1 = ...
```

说明表值会被扩展到更细的小数精度后再插值。

### 13.3 Dithering

当高位宽映射到较低位宽时，直接截断会产生 banding。Dithering 通过加入小幅伪随机或误差反馈扰动，使量化误差空间分散。

源码中的逻辑类似：

```text
out = (y_pos0 << 2) + interpolated + Seed
Seed = out & 0x1f
out = out >> 2
```

这里 `Seed` 取低位反馈到下一个像素，可降低平滑渐变中的条带感。

### 13.4 调参与风险

- LUT 曲线太陡会增加局部对比，但也放大噪声和 banding。
- 高光压缩太强会让画面发灰。
- 暗部提升太多会暴露 black level 和 RAWDNS 残留噪声。
- GTM 是全局曲线，不能根据局部区域单独调整，这正是后续 LTM 的作用。

## 14. LTM：局部色调映射

### 14.1 模块位置与目的

LTM 是 Local Tone Mapping，用局部上下文调整亮度。相比 GTM，LTM 能在保留局部细节的同时压缩大动态范围，例如提亮暗部、压住亮部、增强局部对比。

源码 `ltm.cpp` 显示该模块使用 log-luminance 和 9x9 bilateral filter。

### 14.2 亮度估计

源码中的 luma 近似为：

```text
L = (84*R + 168*G + 4*B) >> 8
```

权重约等于：

```text
R: 84/256  = 0.328
G: 168/256 = 0.656
B: 4/256   = 0.016
```

这比标准 BT.601 更强调 G，弱化 B，适合用作 ISP 内部亮度基准。

### 14.3 Log 域处理

LTM 对亮度取 log：

```text
logL = log10(L)
```

源码使用 `log10tab` 查表，避免硬件中计算对数。

Log 域有两个好处：

- 乘性光照变化变成加性变化。
- 更符合人眼对亮度的感知。

### 14.4 9x9 bilateral filter

LTM 需要把亮度分成 base layer 和 detail layer：

```text
base   = bilateral_filter(logL)
detail = logL - base
```

Bilateral filter 权重由空间距离和亮度差共同决定：

```text
w(i,j) = w_s(i,j) * w_r(|logL(i,j) - logL(center)|)
```

其中：

- `w_s` 是空间核，源码中使用 `pos_kerTab`。
- `w_r` 是 range kernel，源码中使用 `exptab[abs(diff)]`。

加权平均：

```text
base = sum(w(i,j) * logL(i,j)) / sum(w(i,j))
```

源码近似：

```text
weight = (exptab[abs(diff)] * pos_kerTab[i][j]) >> 12
```

### 14.5 Base/detail 合成

得到 base 和 detail 后，局部 tone mapping 可写为：

```text
out_logL = contrast * base + detail
```

如果 `contrast < 1`，base 动态范围被压缩；detail 保留，使局部纹理不被压平。

然后用指数表恢复线性亮度：

```text
out_L = 10^(out_logL)
```

源码使用 `exp10tab` 查表。

### 14.6 RGB 重缩放

为了保持颜色比例，LTM 不直接重新算 RGB，而是用亮度比例缩放原始 RGB：

```text
ratio = out_L / L
R' = R * ratio
G' = G * ratio
B' = B * ratio
```

源码逻辑中有类似 `u_complayer / l_center` 的比例计算，并对 14 bit 输出限幅。

### 14.7 实现骨架

```cpp
for each RGB14 pixel:
  L = (84*R + 168*G + 4*B) >> 8
  log_center = log10tab[L]

  sum_w = 0
  sum_base = 0
  for i,j in 9x9:
    log_neighbor = log10tab[L(i,j)]
    range_w = exptab[abs(log_neighbor - log_center)]
    space_w = pos_kerTab[i][j]
    w = (range_w * space_w) >> 12
    sum_w += w
    sum_base += w * log_neighbor

  base = sum_base / sum_w
  detail = log_center - base
  out_log = contrast * base + detail
  out_L = exp10tab[out_log]

  ratio = out_L / max(L, 1)
  out_rgb = clip14(rgb * ratio)
```

### 14.8 调参与风险

- 空间核太大：局部对比增强范围大，但容易产生 halo。
- range 核太宽：边缘两侧互相影响，halo 更明显。
- contrast 压缩太强：画面局部细节保留，但整体可能发灰。
- 暗部过度提升会放大噪声，所以 LTM 和 RAWDNS/YUVDNS 的强度需要配合。

## 15. CAC：色差校正

### 15.1 模块位置与目的

CAC 用于修正 chromatic aberration。镜头色差会让 R/B 通道相对 G 通道在边缘处错位，表现为紫边、绿边或彩色轮廓。

CAC 位于 LTM 后、CSC 前，在 RGB 域工作。

`cac_register` 包含：

```text
eb          : 使能
t_transient : 中心边缘/瞬态阈值
t_edge      : 稳定边界阈值
```

### 15.2 核心思想

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

### 15.3 7x7 窗口与边缘检测

源码使用 7x7 RGB 窗口。对 R/G/B 分别计算水平和垂直方向边缘，形式接近 Sobel 或差分核：

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

### 15.4 修正选择

源码中不是无条件使用修正值，而会比较修正前后与邻域缓存中的差异，只有满足改善条件时才替换。这样能避免在真实彩色边缘上错误抑制颜色。

### 15.5 实现骨架

```cpp
for each RGB14 pixel with 7x7 window:
  g_edge_h = calc_horizontal_edge(G)
  g_edge_v = calc_vertical_edge(G)

  if max(g_edge_h, g_edge_v) < t_transient:
    out = center
    continue

  direction = g_edge_h > g_edge_v ? horizontal : vertical
  boundary = find_stable_boundary(direction, t_edge)

  rg_low, rg_high = boundary_range(R-G)
  bg_low, bg_high = boundary_range(B-G)

  rg_new = clamp(R_center - G_center, rg_low, rg_high)
  bg_new = clamp(B_center - G_center, bg_low, bg_high)

  R_candidate = G_center + rg_new
  B_candidate = G_center + bg_new

  out = choose_if_improved(candidate, original, local_context)
```

### 15.6 调参与风险

- `t_transient` 太低：很多普通纹理都会被当作色差边缘，颜色被压制。
- `t_transient` 太高：紫边漏检。
- `t_edge` 决定边界搜索的稳定性，过低会找不到边界，过高会把边缘区当成稳定区。
- CAC 不应替代镜头标定；它是局部修正，不是几何重采样级别的完整色差模型。

## 16. CSC：RGB 到 YUV 色彩空间转换

### 16.1 模块位置与目的

CSC 把 RGB 转为 YUV。RGB 适合显示和颜色校正，YUV 更适合视频处理、压缩、亮色分离和后续降噪。

`csc_register` 包含：

```text
eb        : 使能
coeff[12] : 转换矩阵和偏置
```

输出为 `stream_u30`，即 Y/U/V 每通道 10 bit。

### 16.2 标准公式

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

## 17. YFC：YUV 格式转换

### 17.1 模块位置与目的

YFC 将 YUV444 转为 YUV422 或 YUV420。YUV444 每个像素都有 Y/U/V；YUV422 水平方向两个像素共享一组 U/V；YUV420 水平和垂直方向 2x2 像素共享一组 U/V。

`yfc_register` 包含：

```text
eb         : 使能
yuvpattern : 0 = yuv422, 1 = yuv420
```

### 17.2 YUV422

对相邻两个像素：

```text
P0 = (Y0, U0, V0)
P1 = (Y1, U1, V1)
```

输出保留两个亮度，色度取平均：

```text
U01 = (U0 + U1 + 1) / 2
V01 = (V0 + V1 + 1) / 2
```

输出布局可为：

```text
Y0 U01 Y1 V01
```

或内部仍以 stream 形式传递，只是 U/V 被按 pattern 降采样。

### 17.3 YUV420

对 2x2 像素块：

```text
Y00 Y01
Y10 Y11
```

亮度全部保留，色度取 4 个像素平均：

```text
U = (U00 + U01 + U10 + U11 + 2) / 4
V = (V00 + V01 + V10 + V11 + 2) / 4
```

### 17.4 工程注意事项

- 色度下采样应尽量在 CSC 后、YUV 降噪前后根据需求选择。xkISP 主链路中 YFC 在 YUVDNS 前。
- 如果先做 YUV420，再做 YUVDNS，UV 平面分辨率降低，UV 降噪窗口和边界要按 pattern 处理。
- 奇数宽高需要定义边界行为，例如复制最后一列/行或丢弃。

## 18. YUVDNS：YUV 域非局部均值降噪

### 18.1 模块位置与目的

YUVDNS 在 CSC/YFC 后处理 YUV 图像。相比 RAWDNS，它工作在更接近视觉和编码的域：

- Y 通道对应亮度噪声。
- U/V 通道对应色度噪声。

色度噪声通常比亮度噪声更容易被人眼接受更强的平滑，因此 Y 和 UV 使用不同参数。

### 18.2 关键参数

`yuvdns_register` 包含：

```text
eb          : 使能
y_sigma2    : Y 噪声方差
uv_sigma2   : UV 噪声方差
y_invsigma2 : 1 / y_sigma2
uv_invsigma2: 1 / uv_sigma2
y_filter    : Y 滤波强度
uv_filter   : UV 滤波强度
y_invfilter : Y filter 倒数
uv_invfilter: UV filter 倒数
yH2         : Y h^2
uvH2        : UV h^2
yinvH2      : 1 / Y h^2
uvinvH2     : 1 / UV h^2
```

### 18.3 数学原理

仍然是 NLM：

```text
Y'(p) = sum_q w_Y(p,q) Y(q)
U'(p) = sum_q w_UV(p,q) U(q)
V'(p) = sum_q w_UV(p,q) V(q)
```

权重：

```text
w_Y(p,q) = exp(-D_Y(p,q) / h_Y^2)
w_UV(p,q) = exp(-D_UV(p,q) / h_UV^2)
```

距离可以分别基于 Y patch 和 UV patch：

```text
D_Y = sum (Y(p+r) - Y(q+r))^2
D_UV = sum [(U(p+r)-U(q+r))^2 + (V(p+r)-V(q+r))^2]
```

### 18.4 与 RAWDNS 的差异

RAWDNS：

- 工作在 CFA 采样结构中。
- 必须注意同色采样位置。
- 主要处理 sensor 原始噪声。

YUVDNS：

- 工作在完整 YUV 图像或子采样 YUV 图像中。
- 可按人眼感知分开处理亮度和色度。
- 适合抑制 demosaic、sharpen、tone mapping 后显露出的残余噪声。

### 18.5 调参与风险

- Y 降噪太强会让纹理变塑料。
- UV 降噪可适当强一些，但过强会导致颜色溢出边缘或低饱和区域发灰。
- 如果 YFC 已经转为 420，UV 的空间坐标和 Y 不同，窗口对齐要格外注意。

## 19. Scaledown：缩小

### 19.1 模块位置与目的

Scaledown 用于按整数倍缩小图像。它位于 YUVDNS 后，说明降噪先在较高分辨率上完成，再缩小输出。

`scaledown_register` 包含：

```text
eb         : 使能
yuvpattern : YUV 格式
times      : 缩小倍数
```

### 19.2 数学原理

最直接的缩小方式是 block average。若缩小倍数为 `N`，输出像素为输入中 `N x N` 区域平均：

```text
O(x,y) = 1/(N^2) * sum_{i=0}^{N-1} sum_{j=0}^{N-1}
         I(Nx+i, Ny+j)
```

对 Y/U/V 分别执行。若 YUV 是 422 或 420，UV 平面的采样密度不同，需要按实际 chroma layout 处理。

### 19.3 实现骨架

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

### 19.4 工程注意事项

- 直接抽点会 alias，平均池化能降低混叠，但不是高质量重采样滤波。
- `times` 应与输出尺寸整除关系匹配，否则边界需要特殊处理。
- 对 YUV420 数据缩放时，UV 平面本来已降采样，不能简单按 Y 平面的坐标重复处理。

## 20. Crop：裁剪

### 20.1 模块位置与目的

Crop 是主链路末端模块，按 ROI 输出指定矩形区域。

`crop_register` 包含：

```text
eb             : 使能
upper_left_x   : 左上角 x
upper_left_y   : 左上角 y
lower_right_x  : 右下角 x
lower_right_y  : 右下角 y
yuvpattern     : YUV 格式
```

### 20.2 数学定义

给定 ROI：

```text
x0 = upper_left_x
y0 = upper_left_y
x1 = lower_right_x
y1 = lower_right_y
```

输出所有满足条件的像素：

```text
x0 <= x < x1
y0 <= y < y1
```

或如果代码按闭区间处理，则为：

```text
x0 <= x <= x1
y0 <= y <= y1
```

具体边界语义要和测试向量保持一致。

### 20.3 实现骨架

```cpp
for y in 0..height-1:
  for x in 0..width-1:
    p = src.read()
    if reg.eb && inside_roi(x, y):
      dst.write(p)
    else if !reg.eb:
      dst.write(p)
```

最终 `store_out` 会把 Y/U/V 写入输出数组或文件。

### 20.4 工程注意事项

- ROI 必须落在 scaledown 后的坐标系中。
- YUV420/422 的 crop 需要注意偶数对齐，否则 chroma 采样点会错位。
- 视频 pipeline 中 crop 参数改变应通过 shadow register 在帧边界生效，避免一帧内撕裂。

## 21. 主链路模块之间的依赖关系

### 21.1 RAW 域模块顺序

```text
TPG -> DGain -> LSC -> DPC -> RAWDNS -> AWB -> WBC -> GB -> Demosaic
```

这个顺序是合理的：

- DGain 先扣黑电平和做基础增益。
- LSC 修正空间亮度/色彩不均，避免影响 AWB 统计。
- DPC 先消除强异常值，避免 RAWDNS 和 demosaic 扩散坏点。
- RAWDNS 在 demosaic 前降噪，保持 CFA 结构。
- AWB 统计出白平衡增益，WBC 应用。
- GB 修正双绿差异，减少 demosaic 伪影。
- Demosaic 最后把 RAW 变成 RGB。

### 21.2 RGB 域模块顺序

```text
Demosaic -> EE -> CMC -> GTM -> LTM -> CAC
```

这段顺序表示：

- EE 先增强 demosaic 后的边缘。
- CMC 把 sensor RGB 校正到目标 RGB。
- GTM 做全局 tone curve。
- LTM 做局部 tone mapping。
- CAC 在 tone 后仍保留 RGB 通道，修正彩边。

一个值得注意的点是 EE 在 CMC 前。这样锐化作用在 sensor RGB 空间，可能和最终视觉颜色空间存在差异。如果实际图像出现彩色边缘增强，可以考虑评估 EE 和 CMC 的先后关系，但这属于架构变更。

### 21.3 YUV 域模块顺序

```text
CSC -> YFC -> YUVDNS -> Scaledown -> Crop
```

这段顺序表示：

- 先转 YUV，便于亮度/色度分离。
- 再做格式转换。
- 在 YUV 域分别降噪。
- 然后缩小和裁剪输出。

如果追求更高质量，常见替代方案是先 YUVDNS 后 YFC，因为 444 色度信息更完整。但当前实现选择先 YFC，可能是为了降低后续处理带宽。

## 22. 典型调试路径

### 22.1 检查 Bayer pattern

先只开 TPG、DGain、Demosaic、CSC，观察彩条颜色是否正确。如果红蓝互换或绿色异常，优先检查：

```text
top_reg.imgPattern
tpg_reg.imgPattern
Demosaic 的 pattern 使用
```

### 22.2 检查黑电平和增益

输入暗场图：

- DGain 后暗场均值应接近 `top_reg.blc`。
- 如果暗场有明显彩色偏移，检查四通道 BLC。
- 如果暗场被截断成大片 0，说明黑电平扣得过大。

### 22.3 检查 LSC

输入均匀光源图：

- LSC 前应能看到暗角。
- LSC 后中心和四角亮度更一致。
- 如果四角噪声明显，说明增益过大或标定光照不足。

### 22.4 检查 DPC 和 RAWDNS

输入暗场和高 ISO 图：

- DPC 应消除孤立 hot pixel。
- RAWDNS 应降低随机噪声但保留边缘。
- 如果细节明显变糊，先降低 RAWDNS，再检查 DPC 阈值是否误杀纹理。

### 22.5 检查 AWB/WBC

输入灰卡或标准色卡：

- AWB 输出增益应稳定。
- WBC 后灰卡 R/G/B 接近。
- 如果视频中白平衡跳动，应加入帧间平滑或统计区域限制。

### 22.6 检查 Demosaic

输入斜边、棋盘格和高频纹理：

- 看是否有 zipper。
- 看是否有 false color。
- 看绿色通道是否有 maze pattern。

问题定位顺序：

```text
GB -> RAWDNS -> Demosaic 方向判定 -> EE
```

### 22.7 检查 Tone Mapping

输入高动态范围场景：

- GTM 决定整体曲线。
- LTM 决定局部亮度和 halo。
- 如果暗部噪声严重，先降低 LTM 暗部提升或增强前级降噪。

### 22.8 检查 YUV 输出

灰阶图应满足：

```text
U ~= neutral
V ~= neutral
```

如果灰阶偏色，检查 CSC offset 和系数。若 YUV420/422 输出色彩错位，检查 YFC、Scaledown、Crop 的偶数对齐。

## 23. 附录：仓库中存在但未接入主链路的模块

### 23.1 AEC：自动曝光统计

`aec.cpp` 中可见 AEC 计算加权亮度：

```text
Y = (2R + 9G + 5B + 8) >> 4
```

权重大约为：

```text
R: 0.125
G: 0.5625
B: 0.3125
```

然后按亮度区间分类，例如黑区、中间区、高亮区，并累积加权均值，输出 `mean_y`。该结果可给 AE 控制环使用，用于调整 exposure time、analog gain、digital gain。

AEC 不在当前 `top.cpp` 主链路中直接调用，因此本文不把它列入主图像处理路径。

### 23.2 AFC：自动对焦统计

`afc.cpp` 中可见 AFC 在指定窗口内计算 focus value。窗口大小约为 128x128，位置由 `location_row/col` 指定。它使用 Sobel、Laplace、H5/H9 等高频滤波器，统计高频和中频能量。

对焦评价函数的基本思想是：

```text
focus_value = sum(abs(high_frequency_response))
```

图像越清晰，边缘和纹理越强，高频响应越大。AF 控制器可移动 lens position，寻找 focus value 最大的位置。

### 23.3 HistEQ：直方图均衡

文档中有 HistEQ，但主 `top.cpp` 链路未调用。其数学基础是：

```text
PDF(i) = n(i) / N
CDF(I) = sum_{i=0}^{I} PDF(i)
```

映射函数：

```text
f(x) = round((L - 1) * CDF(x))
```

其中 `L` 是灰阶级数。直方图均衡会增强全局对比度，但可能放大噪声，并改变整体 tone 风格。当前链路使用 GTM/LTM 作为 tone mapping 主方案。

## 24. 硬件设计与仿真实现

前面的章节主要从算法角度解释每个模块。这个工程另一个重要特点是：它不是单纯的软件 ISP，而是以 HLS/RTL 落地为目标写的。代码中同时存在 Vivado HLS、Catapult HLS、软件 testbench、FPGA host 和外部 RTL wrapper。理解这部分，对真正掌握 ISP 硬件实现很关键。

### 24.1 工程中的硬件相关目录

硬件实现相关文件主要分布如下：

```text
src/
  HLS C++ 主算法。通过 #ifdef vivado / #ifdef catapult 兼容不同 HLS 工具。

tb/
  Vivado HLS C 仿真 testbench。每个模块有独立 tb，top 也有完整 tb_top.cpp。

tcl/
  Vivado HLS 工程脚本和 directives。包括 top.tcl、top_directives.tcl 和各模块 directives。

catapult/
  Catapult HLS 版本的工程脚本、配置、testbench、部分源码副本和生成结果。

isp_itf/
  手写或集成用 RTL wrapper。包括 AXI-Lite 寄存器、AXI master 读输入、输入 FIFO、HLS IP 包装。

fpga/
  Xilinx OpenCL/XRT 风格 host/kernel 示例，用于把 HLS kernel 接到 FPGA runtime。

tv/
  test vector、输入 raw、配置文件、golden 输出。
```

所以它的验证链路大致是：

```text
C/C++ 算法仿真
  -> HLS 综合
  -> HLS C/RTL cosim
  -> 导出 IP
  -> RTL wrapper 接 AXI/AXI-Lite/FIFO
  -> FPGA host 或系统级集成
```

### 24.2 顶层 HLS 接口

`src/top.cpp` 中顶层函数是：

```cpp
void isp_top(
    stream_u12 &src,
    uint16* y_ptr,
    uint16* u_ptr,
    uint16* v_ptr,
    top_register top_reg,
    tpg_register tpg_reg,
    dgain_register dgain_reg,
    ...
    crop_register crop_reg
)
```

接口可以分成三类：

```text
src              : 输入 RAW stream，uint12 像素流
y_ptr/u_ptr/v_ptr: 输出 Y/U/V 写回 DDR 的指针
*_register       : 模块配置寄存器
```

Vivado HLS pragma 中明确指定：

```cpp
#pragma HLS dataflow
#pragma HLS INTERFACE m_axi depth=307200 port=v_ptr offset=direct bundle=gmem2
#pragma HLS INTERFACE m_axi depth=307200 port=u_ptr offset=direct bundle=gmem1
#pragma HLS INTERFACE m_axi depth=307200 port=y_ptr offset=direct bundle=gmem0
#pragma HLS STREAM variable=src dim=1
```

这意味着：

- 输入不是通过 HLS 自己的 `m_axi` 读 DDR，而是外部把 RAW 喂成 stream。
- 输出使用三个独立 AXI master bundle：`gmem0/gmem1/gmem2`，分别写 Y/U/V。
- 三个输出通道拆成独立 bundle 可以并行写 DDR，避免 Y/U/V 共用一个 AXI 端口造成写带宽瓶颈。
- `offset=direct` 表示指针地址作为直接端口传入，而不是 AXI-Lite 内部寄存器自动管理 offset。

顶层 `dataflow` 是最关键的硬件结构指令。没有它，HLS 可能把模块串行调度成“跑完 TPG 再跑 DGain 再跑 LSC”。有了 `dataflow`，每个模块可以作为并发 process，通过 FIFO/stream 串起来。

### 24.3 主数据流：模块级流水线

`isp_top` 内部定义了大量中间 stream：

```text
tpg_dgain_data      : stream_u12
dgain_lsc_data      : stream_u12
lsc_dpc_data        : stream_u12
dpc_rawdns_data     : stream_u12
rawdns_awb_data     : stream_u12
awb_wbc_data        : stream_u12
wbc_gb_data         : stream_u12
gb_demosaic_data    : stream_u12
demosaic_ee_data    : stream_u36
ee_cmc_data         : stream_u36
cmc_gtm_data        : stream_u42
gtm_ltm_data        : stream_u42
ltm_cac_data        : stream_u42
cac_csc_data        : stream_u42
csc_yfc_data        : stream_u30
yfc_yuvdns_data_y/u/v : stream_u10
yuvdns_scale_data_y/u/v: stream_u10
scale_crop_data_y/u/v : stream_u10
dst_y/u/v              : stream_u10
```

Vivado 版本中，每个 stream 都设置了：

```cpp
#pragma HLS STREAM variable=<name> depth=2 dim=1
```

这说明设计意图不是用大 FIFO 缓冲整帧，而是模块之间只放很浅的弹性 FIFO。真正的图像上下文由各模块自己的 line buffer 保存。

一个像素在硬件中的流动方式是：

```text
src stream 每拍读入一个 RAW12
  -> 模块 A 处理后写入 depth=2 FIFO
  -> 模块 B 同时从 FIFO 读取
  -> 继续向后传播
  -> crop 后拆成 Y/U/V 三个 stream
  -> store_out 每拍读 Y/U/V 并写 DDR
```

理想情况下，稳定态吞吐为：

```text
1 pixel / cycle
```

但真实吞吐会受以下因素影响：

- 最慢模块的 II。
- 边界 padding/flush 产生的额外周期。
- 输出 AXI 写通道是否 backpressure。
- NLM、LTM、CAC 这类复杂模块内部是否能达到 II=1。

### 24.4 为什么中间 FIFO 深度只有 2

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

### 24.5 行缓存与滑动窗口

ISP 硬件里最重要的面积结构是 line buffer。因为输入是按 raster scan 顺序来的：

```text
(0,0), (0,1), ..., (0,W-1),
(1,0), (1,1), ...
```

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

这里 `8192` 是最大行宽上限，和 `top_directives.tcl` 中 loop tripcount `max=8192` 对齐。LTM 的 line buffer 使用 `[8][4096]`，说明该模块按更小最大宽度或不同处理假设配置。

### 24.6 窗口数组为什么 complete partition

HLS directives 中大量出现：

```tcl
set_directive_array_partition -type complete -dim 0 "..." window
set_directive_unroll "..." window_loop
```

窗口数组如 `rawWindow[5][5]`、`rgbWindow[7][7]`、`yWindow[9][9]` 会被完全分割成寄存器。这样一个周期内可以并行读取窗口内多个像素。

如果窗口仍然是普通 RAM，一个双口 RAM 每拍最多读两个值。但 5x5/7x7/9x9 算法每拍可能需要读几十个窗口元素：

```text
5x5  = 25 taps
7x7  = 49 taps
9x9  = 81 taps
```

不 partition 的结果通常是：

- HLS 因读端口不足无法 II=1。
- 或者复制 RAM，面积不可控。
- 或者拉长调度，吞吐下降。

所以该工程的面积策略是：

```text
小窗口 complete partition 成寄存器，换吞吐。
大行缓存 block partition，控制 RAM 端口和面积。
```

### 24.7 行缓存为什么 block partition

对 line buffer，directives 常见：

```tcl
set_directive_array_partition -type block -factor 4  -dim 1 "dpc" lineBuffer
set_directive_array_partition -type block -factor 10 -dim 1 "isp_rawdns" rawdns_lines
set_directive_array_partition -type block -factor 8  -dim 1 "ltm" rlineBuf
```

line buffer 是大数组，例如：

```text
rawdns_lines[10][8192]，uint12
```

总 bit 数约：

```text
10 * 8192 * 12 = 983040 bit
```

这显然应映射到 BRAM/URAM，而不是 complete partition 成寄存器。block partition 的目的通常是：

- 把多行拆成多个 memory bank。
- 为同一周期内多行读写提供更多端口。
- 避免 complete partition 导致寄存器面积爆炸。

这也是硬件 ISP 的基本取舍：

```text
窗口寄存器化，行缓存 RAM 化。
```

### 24.8 边界延迟与输出对齐

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

### 24.9 定点化策略

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

### 24.10 统计类模块如何硬件化

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

### 24.11 面积节省手段

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

### 24.12 吞吐优化手段

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

### 24.13 RTL 外部接口包装

`isp_itf/isp_top_wrap.v` 是系统集成层。它把 HLS 生成的 `isp_top` 包在更完整的接口中，包括：

```text
S_AXIL_*      : AXI-Lite 配置寄存器接口
M_AXI_*       : RAW 输入 DDR 读取 master
m_axi_gmem0   : Y 输出 DDR 写 master
m_axi_gmem1   : U 输出 DDR 写 master
m_axi_gmem2   : V 输出 DDR 写 master
fifo_in       : 输入 FIFO，64 bit 写入，16 bit 读出
ap_start/done/idle/ready : HLS IP 控制信号
```

外部 RAW 读路径大致为：

```text
AXI master 从 DDR burst read 64-bit 数据
  -> fifo_in 写入 64-bit word
  -> fifo_in 每次读出 16-bit
  -> 接到 HLS src_V_V_dout
  -> HLS 内部使用低 12 bit RAW
```

`fifo_in` 参数：

```verilog
FIFO_WDATA_WIDTH = 64
FIFO_RDATA_WIDTH = 16
FIFO_DEPTH       = 4096
```

这说明外部带宽按 64 bit burst 吃 DDR，内部按像素流 16 bit 供给 HLS。12 bit RAW 被装在 16 bit 容器中是常见做法，简化地址对齐和 AXI 数据打包。

输出路径则由 HLS `m_axi` 端口直接写 Y/U/V 三个 plane。`store_out` 每个输出像素从三个 `stream_u10` 读出，然后写：

```cpp
y_ptr[i] = temp_y;
u_ptr[i] = temp_u;
v_ptr[i] = temp_v;
```

同时有：

```cpp
#pragma HLS DEPENDENCE variable=y_ptr inter false
#pragma HLS PIPELINE
```

这表示 store 阶段也尝试做到连续写，不让 HLS 误判指针写存在循环相关。

### 24.14 AXI-Lite 寄存器映射

`isp_itf/isp_saxil.v` 中生成了大量寄存器输出，例如：

```text
top_register_frameWidth
top_register_frameHeight
top_register_imgPattern
dgain_register_m_nR
wbc_register_m_nB
lsc_register_rGain
cmc_register_m_nGain
gtm_register_gtmTab
csc_register_coeff
crop_register_upper_left_x
...
```

这些寄存器通过 AXI-Lite 写入，再作为 direct signal 连接到 HLS IP。也就是说，软件侧配置流程是：

```text
CPU/host 写 AXI-Lite 寄存器
  -> wrapper 输出配置线
  -> HLS isp_top 的 register struct 字段
  -> 每个模块读取自己的配置
```

`isp_top_wrap.v` 中还有一个小状态机处理 `ap_start`：

```text
IDLE -> START -> RUN -> IDLE
```

注释指出：

```text
ap_start must hold high until ap_ready=1
```

这符合 HLS ap_ctrl_hs 控制协议。系统集成时必须保证开始信号保持到 IP 接受，否则一帧可能没有启动。

### 24.15 Vivado HLS 仿真流程

`tcl/top.tcl` 指定：

```tcl
open_project top
set_top isp_top
add_files ../src/*.cpp
add_files -tb ../tb/tb_top.cpp
open_solution "top"
source "./top_directives.tcl"
source "./script.tcl"
```

`script.tcl` 中：

```tcl
set_part {xcvu9p-fsgd2104-2-e}
create_clock -period 20 -name default
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
exit
```

这表示默认 Vivado HLS 目标器件是 Xilinx VU9P，时钟周期 20 ns，即 50 MHz。流程包括：

```text
csim   : C 仿真，验证 C++ 算法和 testbench
csynth : HLS 综合，生成 RTL 和资源/时序报告
cosim  : C/RTL 协同仿真，验证 RTL 和 C 行为一致
export : 导出 IP catalog
```

### 24.16 tb_top.cpp 的验证方式

`tb/tb_top.cpp` 做了完整 top 级仿真：

```text
1. 声明所有 register struct
2. 从 tv/hls_param.txt 读取配置
3. malloc 输入、golden、输出 buffer
4. 从 tv/input.raw 读取 uint16 RAW
5. 把每个 RAW 转成 uint12 写入 src stream
6. 调用 isp_top(src, dst_y, dst_u, dst_v, ...)
7. 把输出 Y/U/V 重新打包成 frameOut
8. 和 tv/output.yuv 逐点比较
9. 遇到第一处 mismatch 输出 pixel/channel/golden/result
```

关键代码逻辑可概括为：

```cpp
for x in frameWidth * frameHeight:
    fread(&frameIn[x], sizeof(uint16_t), 1, fp_r1);
    src.write((uint12)frameIn[x]);

isp_top(src, dst_y, dst_u, dst_v, ...);

for x in 3 * output_width * output_height:
    if (frameGolden[x] != frameOut[x])
        report first mismatch
```

这类 testbench 对硬件开发非常重要，因为它验证的是 bit-exact 行为。ISP 算法软件版如果用 float，而 HLS 用定点，通常会出现误差；这里 testbench 直接比较整数输出，说明 golden 应该按同一套定点模型生成。

### 24.17 单模块 testbench 的意义

`tb/` 目录下每个主模块都有单独 testbench，例如：

```text
tb_dgain.cpp
tb_lsc.cpp
tb_dpc.cpp
tb_rawdns.cpp
tb_demosaic.cpp
tb_ee.cpp
tb_cmc.cpp
tb_gtm.cpp
tb_ltm.cpp
tb_cac.cpp
tb_csc.cpp
tb_yfc.cpp
tb_yuvdns.cpp
tb_scaledown.cpp
tb_crop.cpp
```

单模块仿真的价值是：

- 快速定位某一级的 bit-exact mismatch。
- 缩短 C/RTL cosim 时间。
- 让模块可以独立综合，看单模块 II、latency、LUT/FF/BRAM/DSP。
- 调参时只更新相关 test vector，不必每次跑完整 ISP。

对 ISP 硬件项目来说，推荐验证顺序是：

```text
模块 C sim
  -> 模块 C/RTL cosim
  -> top C sim
  -> top C/RTL cosim
  -> wrapper/system sim
  -> FPGA board test
```

### 24.18 Catapult HLS 路径

工程还提供了 Catapult 版本：

```text
catapult/top_tcl/hls_isp.tcl
catapult/config/xkISP_HLS.cfg
catapult/module_tcl/
catapult/module_dv/
catapult/tb/
catapult/src/
```

Catapult 配置中使用：

```tcl
solution options set /Input/CppStandard c++11
directive set -DESIGN_GOAL area
solution design set isp_top -top
solution design set tpg -block
solution design set dgain -block
...
solution library add nangate-45nm_beh
directive set -CLOCKS {clk {-CLOCK_PERIOD 10.0 ...}}
```

这表示 Catapult 路径目标更偏 ASIC/通用 HLS，而不只是 Xilinx FPGA。源码中 `top.h` 通过：

```cpp
#ifdef vivado
  typedef ap_uint<...>
  typedef hls::stream<...>
#endif

#ifdef catapult
  typedef ac_int<...>
  typedef ac_channel<...>
#endif
```

把同一套算法映射到不同 HLS 类型系统。

设计上需要注意：

- Vivado `hls::stream` 和 Catapult `ac_channel` 都是 FIFO/channel 抽象，但调度和接口细节不同。
- `#pragma HLS` 和 Catapult `directive set` 不完全等价。
- 同一 C++ 代码在两个工具下综合结果可能不同，必须分别做 cosim。

### 24.19 FPGA host/kernel 示例

`fpga/host.cpp` 是 Xilinx XRT/OpenCL 风格 host 示例。它做的事情包括：

```text
读取 tv/input.raw
读取 tv/hls_param.txt
打包配置结构
创建 device buffer
启动 kernel
读回 output.yuv
```

这条路径更像是把 ISP 当作加速 kernel 使用，而 `isp_itf/` 更像是 SoC/RTL 系统集成方式。两者关注点不同：

```text
fpga/host.cpp : 软件 runtime 调 kernel
isp_itf/*.v   : RTL 顶层接 AXI 总线和寄存器
```

如果要深入做 FPGA 板级验证，重点应先看 `isp_itf`，因为它更接近真实硬件数据通路：

```text
DDR RAW -> AXI read master -> FIFO -> HLS stream -> HLS pipeline -> AXI write Y/U/V
```

### 24.20 按模块看硬件复杂度

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

如果资源不够，优先考虑：

- 降低搜索窗口大小。
- 降低并行 unroll 因子。
- 减少 NLM 权重候选数。
- 使用更粗 LUT 或分段线性近似。
- 降低中间位宽。
- 让 RAWDNS 和 YUVDNS 二选一，而不是都开。

### 24.21 这套硬件设计的核心取舍

这套代码的硬件设计可以概括为五个取舍。

第一，吞吐优先：

```text
dataflow + pipeline loop + stream 串接
```

目标是整条链路流起来，而不是模块级批处理。

第二，局部缓存替代整帧缓存：

```text
KxK 算法只保存 K-1 行
```

这极大降低外部 DDR 带宽需求，也减少延迟。

第三，窗口并行换 II：

```text
window complete partition + inner loop unroll
```

代价是 LUT/FF 增加。

第四，大缓存 RAM 化：

```text
line buffer block partition
```

在端口数和 BRAM 面积之间折中。

第五，数学函数查表化：

```text
exp/log/gamma/weight -> LUT + fixed-point
```

用 ROM/寄存器和简单乘加替代复杂运算单元。

### 24.22 阅读这套代码时建议关注的硬件问题

如果下一步要继续深入，可以按下面问题读源码和 HLS report：

```text
1. 每个模块综合后的 II 是多少？是否全部 II=1？
2. 哪些模块占 BRAM 最多？一般会是 RAWDNS/LTM/YUVDNS/CAC。
3. 哪些模块占 DSP 最多？一般是 CMC/CSC/LTM/NLM/LSC。
4. 哪些数组被映射成 register，哪些映射成 RAM？
5. line buffer 是否存在读写同地址冲突？
6. dependence false 是否真的安全？
7. 输出 AXI 写带宽是否足够承受 3 plane 同时写？
8. 输入 FIFO 深度 4096 对 DDR burst 抖动是否足够？
9. crop/scaledown/yuv420 场景下输出像素数是否和 store_out 完全一致？
10. shadow register 是否保证参数在帧边界更新？
```

这几个问题比单看算法更接近真实硬件 ISP 的风险点。

## 25. 总结

xkISP 主模块是一条完整的 RAW-to-YUV ISP pipeline。它的结构特点是：

- RAW 前端重视传感器物理问题：黑电平、镜头阴影、坏点、RAW 噪声、双绿不平衡。
- 中段用边缘自适应 demosaic 恢复 RGB，再通过锐化、色彩矩阵和 tone mapping 形成视觉图像。
- 后段在 YUV 域做格式转换、视觉降噪、缩放和裁剪，面向编码或显示输出。
- 工程实现大量使用定点乘加、查表、移位、clip、line buffer 和 sliding window，符合 FPGA/HLS 的吞吐与资源约束。
- 硬件结构上采用 `dataflow + stream + 浅 FIFO + 行缓存 + 滑动窗口`，目标是逐像素流式处理，而不是每级写回整帧。
- 仿真体系比较完整，既有单模块 testbench，也有 top 级 bit-exact testbench，还有 Vivado/Catapult 两套 HLS 脚本和外部 AXI wrapper。

从算法调试角度，最重要的是按链路顺序逐级验证，不要直接看最终 YUV 图定位问题。RAW 域错误会被 demosaic 和后续增强放大；色彩矩阵和 tone mapping 的错误会掩盖前级问题；YUV 格式和 crop/scaler 的坐标问题则常表现为色度错位或边界异常。逐模块导出中间结果，是调通该工程最有效的方法。

从硬件调试角度，最重要的是同时看三类结果：C 仿真是否 bit-exact、C/RTL cosim 是否一致、HLS report 中 II/latency/BRAM/DSP 是否符合预期。ISP 设计的难点不只是算法正确，而是算法能否在固定吞吐、固定位宽、有限缓存和有限面积下持续稳定地流动。
