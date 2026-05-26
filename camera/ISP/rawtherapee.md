# 前处理模块

## 2. 主要流程

`RawImageSource::preprocess()` 的顺序大致是：

```text
1. 计算 preprocess WB / scale_mul 参考
2. 查找 dark frame
3. 标记 zero bad pixels
4. 查找 flat field
5. copyOriginalPixels(): 复制 RAW、扣 dark、应用 flat field
6. DNG gain map
7. 静态 badpixels 文件
8. dark frame hot pixels
9. scaleColors(): black / white / WB raw scale
10. lens profile vignetting
11. 动态 hot/dead pixel 检测
12. PDAF line 标记 (依赖配置文件)
13. green equilibration
14. bad pixel interpolation
15. CFA line denoise
16. RAW CA correction
17. 为 denoise 计算 auto exposure histogram
```

这个顺序很重要：坏点先标记，绿平衡和 PDAF 可以继续添加坏点或修正 RAW，最后统一插值修复，再进行行噪声和 CA。

## 4. Dark frame

```text
减暗电流
抑制长曝光热噪声
提供 hot pixel map
```

## 5. Flat field / LSC / Dust correction

```text
镜头暗角校正
传感器响应不均匀校正
灰尘阴影校正
通道响应校正
```

从 ISP 角度看，它接近 lens shading correction，但数据来自用户 flat field 或元数据，而不是相机 ISP 标定表。

## 6. DNG gain map

如果 RAW 文件中有 DNG gain map，并且 `raw.ff_FromMetaData` 启用，RawTherapee 会把 gain map 应用到 RAW 数据上。

这部分比通用 flat field 更接近 DNG 标准里的 LSC/gain compensation。

## 7. DPC：坏点、热坏点、死点

```text
1. detection / marking：把坏点坐标写入 PixelsMap
2. correction：根据 sensor pattern 选择插值函数统一修复
```

### 7.2 坏点来源

RawTherapee 的坏点来源有五类。

第一类是 zero pixels：

算法很直接：扫描 `ri->data`，凡是原始值等于 0 的像素，都写入 `PixelsMap`。这类点主要来自某些相机/RAW 格式中用 0 表示无效像素或 DNG opcode 里的 fixed bad pixel constant。

第二类是 `.badpixels` 静态坏点表：

这类坏点表适合修正固定传感器缺陷。

第三类是 dark frame 提取的 hot pixels：

提取逻辑是：

```text
m = 周围 8 个同 CFA 相位邻居之和
if center > m * (10 / 8):
    标记为 hot pixel
```

这里的 8 个邻居是同色采样位置：

```text
(row-2,col-2) (row-2,col) (row-2,col+2)
(row,  col-2)               (row,  col+2)
(row+2,col-2) (row+2,col) (row+2,col+2)
```

第四类是主图动态 hot/dead pixel 检测：

第五类是 PDAF line 标记， PDAF 异常点也写入同一个 `PixelsMap`，最后复用 DPC 插值。

显然这里不能检测大面积坏点簇
### 7.3 动态 hot/dead 检测算法

`findHotDeadPixels()` 的核心思想是：先在同 CFA 相位邻域上构造一个局部“正常值”估计，再看中心像素相对这个估计是否是局部离群点。它来自 Emil Martinec 的思路，代码里做了 OpenMP 和 SSE 优化。

第一步，对每个候选像素 `(i, j)`，在同色 3x3 采样网格上取中值：

```text
median(
  raw[i-2][j-2], raw[i-2][j], raw[i-2][j+2],
  raw[i  ][j-2], raw[i  ][j], raw[i  ][j+2],
  raw[i+2][j-2], raw[i+2][j], raw[i+2][j+2]
)
```

这个 3x3 不是连续 3x3，而是步长为 2 的同 CFA 相位窗口。这样检测只比较同色像素，不会把 R/G/B 响应差异误判成坏点。

第二步，计算中心像素相对中值的残差：

```text
cfablur[i][j] = rawData[i][j] - median_same_color_3x3(i, j)
```

如果只检测 hot pixel，就忽略负残差；如果只检测 dead pixel，就忽略正残差：

```text
if !findDeadPixels && pixdev <= 0: continue
if !findHotPixels  && pixdev >= 0: continue
```

因此：

```text
pixdev > 0：中心比同色局部中值亮，候选 hot pixel
pixdev < 0：中心比同色局部中值暗，候选 dead pixel
```

第三步，用残差图的 5x5 局部能量作为自适应阈值参考：

```text
pixdev_abs = abs(cfablur[i][j])
hfnbrave = sum(abs(cfablur[5x5 around i,j])) - pixdev_abs
```

代码里实现为：

```text
hfnbrave = -pixdev_abs
sum5x5(cfablur, cc - 2, hfnbrave)
```

也就是先减掉中心残差，再累加周围 5x5 残差绝对值，得到“邻域其它残差总量”。如果周围本来就是强边缘/强纹理，`hfnbrave` 会变大，检测阈值随之提高；如果周围很平坦，轻微孤立异常也更容易被抓到。

第四步，用用户阈值换算成内部阈值：

```text
varthresh = (20 * (thresh / 100) + 1) / 24
```

最终判定：

```text
if abs(center_residual) > varthresh * neighbor_residual_sum:
    bpMap.set(j, i)
```

这个判据的含义是：中心残差必须相对邻域其它残差总和足够突出，才算坏点。它不是单纯的固定亮度阈值，所以对曝光、ISO、局部纹理有一定自适应能力。

边界处理上，动态检测只处理：

```text
row: 2 .. H-3
col: 2 .. W-3
```

因为它需要同色 3x3 和残差 5x5 邻域。边缘 2 像素不参与动态检测。

### 7.4 Bayer 插值算法

`interpolateBadPixelsBayer()` 的核心不是简单均值，而是“同色、成对、梯度加权”的插值。

对每个被 `PixelsMap` 标记的坏点，先跳过边界 2 像素，只处理：

```text
row: 2 .. H-3
col: 2 .. W-3
```

然后只使用同 CFA 相位的有效邻居。每个候选方向都取一对位于坏点两侧的像素，如果这对像素中任意一个也被标记为坏点，该方向直接不用。

对绿色坏点，额外使用两个近对角方向：

```text
(row-1, col-1) + (row+1, col+1)
(row-1, col+1) + (row+1, col-1)
```

距离权重是：

```text
0.70710678 = 1 / sqrt(2)
```

对红/蓝坏点，对角同色点更远，使用：

```text
(row-2, col-2) + (row+2, col+2)
(row-2, col+2) + (row+2, col-2)
```

距离权重是：

```text
0.35355339 = 1 / sqrt(8)
```

所有通道都会再尝试水平和垂直同色方向：

```text
(row,   col-2) + (row,   col+2)   weight = 0.5 = 1 / 2
(row-2, col  ) + (row+2, col  )   weight = 0.5 = 1 / 2
```

每个方向的权重形式是：
pixel_a - pixel_b 差异大，对应权重缩小，即可能是跨过边缘
```text
dirwt = distance_weight / (abs(pixel_a - pixel_b) + eps)
eps = 1
```

累加方式：

```text
wtdsum += dirwt * (pixel_a + pixel_b)
norm   += dirwt
```

最后：

```text
rawData[row][col] = wtdsum / (2 * norm)
```

这里除以 `2 * norm` 是因为每个方向加入的是一对像素的和。算法偏向选择两侧值接近的方向；如果某方向跨过边缘，两侧差值大，权重就小。这样可以减少坏点插值造成的 zipper、亮暗边缘拖影和假色。

如果所有方向都不可用，也就是坏点周围有效同色成对邻居全部被坏点 map 排除，则退回备用方案：

```text
在 5x5 同相位位置上收集未标记坏点
rawData[row][col] = simple_average(valid_same_phase_pixels)
```

代码注释认为能走到备用路径的概率很低，主路径通常能找到至少一对有效像素。


## 9. Green equilibration

### 全局绿平衡

全局绿平衡就是统计全局的Gr Gb差异，用固定修正量全局修正

```text
sum_g1, count_g1 = 0
sum_g2, count_g2 = 0

// 统计全局 Gr Gb
for each green pixel inside border:
    if row parity belongs to G1:
        sum_g1 += value
        count_g1++
    else:
        sum_g2 += value
        count_g2++

mean_g1 = sum_g1 / count_g1
mean_g2 = sum_g2 / count_g2
target = (mean_g1 + mean_g2) / 2

// 矫正系数
gain_g1 = target / mean_g1
gain_g2 = target / mean_g2

// Gr Gb 乘对应增益
for each green pixel:
    value *= gain_for_its_green_phase
```

### 局部绿平衡

局部绿平衡来自 Emil Martinec 的 directional average 思路。按局部方向估计当前绿色点应该接近的另一套绿色相位，并在判断不是纹理细节时做温和修正。


```text
copy green samples from rawData to half-width cfa

// 统计局部区域 Gr 跟 Gb
for each candidate green pixel gin:
    o1 = four diagonal green neighbours // 跟当前位置相异的G
    o2 = four axial green neighbours at distance 2 // 跟当前位置相同的G

	// d1 d2 两个组内绿色之和
    d1 = sum(o1)
    d2 = sum(o2)
    // c1 c2 组内不一致性: 组内四个值两两差的绝对值之和
    c1 = pairwise_abs_diff_sum(o1)
    c2 = pairwise_abs_diff_sum(o2)
    tf = thresh(row, col)
    
    if c1 + c2 >= 6 * tf * abs(d1 - d2):
        continue  
    // 这个判断区分纹理区域跟平坦区域
	// c1 + c2 小：两组邻域各自比较平滑，不像复杂纹理
	// abs(d1 - d2) 大：两组绿色估计有明显差异,可能不平衡
	// 边缘区域 c1 + c2 变大，不会绿平衡

	// 绿通道插值
    estimate gse, gnw, gne, gsw
    compute directional weights wtse, wtnw, wtne, wtsw
    ginterp = weighted_average(direction estimates)

    if ginterp - gin < tf * (ginterp + gin):
        rawData[row][col] = 0.5 * (gin + ginterp)
```


**方向插值估计：ginterp**

通过第一道判定后，算法估计当前绿色点 `gin` 应该接近的值。

当前值：

```text
gin = cfa[rr][cc]
```

先构造四个方向的差分：

```text
gmp2p2 = gin - cfa[rr + 2][cc + 2]
gmm2m2 = gin - cfa[rr - 2][cc - 2]
gmm2p2 = gin - cfa[rr - 2][cc + 2]
gmp2m2 = gin - cfa[rr + 2][cc - 2]
```

再构造四个方向估计值（加一半偏差值）：

```text
gse = o1_4 + 0.5 * gmp2p2
gnw = o1_1 + 0.5 * gmm2m2
gne = o1_2 + 0.5 * gmm2p2
gsw = o1_3 + 0.5 * gmp2m2
```

可以理解为：用**对角近邻 `o1` 加上半个远端同向差分**，估计当前点在 SE/NW/NE/SW 四个方向上的合理绿色值。

然后计算方向权重（其中 eps = 1）：

```text
wtse = 1 / (eps + (gin - far_se)^2 + (next_se - o1_4)^2)
wtnw = 1 / (eps + (gin - far_nw)^2 + (next_nw - o1_1)^2)
wtne = 1 / (eps + (gin - far_ne)^2 + (next_ne - o1_2)^2)
wtsw = 1 / (eps + (gin - far_sw)^2 + (next_sw - o1_3)^2)
```

权重越大，说明该方向局部变化越平滑；权重越小，说明该方向可能跨越边缘或纹理。

最后加权平均：

```text
ginterp =
    (gse * wtse + gnw * wtnw + gne * wtne + gsw * wtsw)
    / (wtse + wtnw + wtne + wtsw)
```

这一步和 demosaic 里的方向插值思想很像：让平滑方向贡献更多。

**实际更新规则**

局部算法不是直接把 `gin` 替换成 `ginterp`。它还要过第二个条件：

```text
if ginterp - gin < tf * (ginterp + gin):
    rawData[rr][cc] = 0.5 * (ginterp + gin)
```

更新值是当前值和估计值的平均：

```text
new_g = 0.5 * (gin + ginterp)
```

这说明修正是保守的：即使判定需要绿平衡，也只往估计值方向走一半，而不是完全替换。

这个规则还有一个方向性：

```text
当 ginterp < gin 时，ginterp - gin 为负，通常更容易通过；
当 ginterp > gin 时，需要满足相对差异限制。
```

所以它更谨慎地抬高绿色值，避免在高亮边缘或噪声处制造新的亮点。




## 10. CFA line denoise

相关文件：

```text
cfa_linedn_RT.cc
```

入口：

```text
cfa_linedn()
ddct8x8s()
```

参数：

```text
raw.bayersensor.linenoise
raw.bayersensor.linenoiseDirection
```

支持方向：

```text
horizontal
vertical
both
PDAF_LINES
```

代码里有 8x8 DCT 相关函数，说明它不是普通均值滤波，而是对 CFA 行/列噪声做频域/块处理。

这个模块适合单独深入，尤其是行噪声、列噪声和 PDAF line 的交互。

## 11. RAW CA correction

相关文件：

```text
CA_correct_RT.cc
```

入口：

```text
CA_correct_RT()
```

触发条件：

```text
raw.ca_autocorrect
raw.cared
raw.cablue
```

特点：

```text
发生在 demosaic 前
只针对 Bayer
支持自动估计和手动红/蓝修正
支持避免颜色偏移
```

RAW 域 CA 比 demosaic 后 RGB 拉伸更合理，因为 CFA 上红/蓝采样还没有被插值扩散。

## 12. Lens vignetting

`preprocess()` 中也会在没有 flat field 时使用镜头 profile 的 vignetting correction：

```text
Lensfun
LCP
metadata lens correction
```

相关文件：

```text
rtlensfun.cc
lcp.cc
lensmetadata.cc
iptransform.cc
```

它和 flat field 的区别：

```text
flat field：基于实际拍摄或 DNG gain map
lens profile：基于镜头模型/数据库/元数据
```


