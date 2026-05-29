# 前处理模块

大致顺序：

```text
1. 计算 preprocess WB / scale_mul 参考
2. 查找 dark frame (主要处理暗电流，长曝光热噪声)
3. 标记 zero bad pixels
4. 查找 flat field （镜头暗角校正、传感器响应不均匀校正、灰尘阴影校正、通道响应校正）
5. copyOriginalPixels(): 复制 RAW、扣 dark、应用 flat field
6. DNG gain map（类似LSC的作用）
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

## DPC：坏点、热坏点、死点

```text
1. detection / marking：把坏点坐标写入 PixelsMap
2. correction：根据 sensor pattern 选择插值函数统一修复
```

### 坏点来源

RawTherapee 的坏点来源有五类：

1. zero pixels：
	扫描 `ri->data`，凡是原始值等于 0 的像素，都写入 `PixelsMap`。这类点主要来自某些相机/RAW 格式中用 0 表示无效像素或 DNG opcode 里的 fixed bad pixel constant。
2. `.badpixels` 静态坏点表
3.  dark frame 提取的 hot pixels：
```text
m = 周围 8 个同 CFA 相位邻居之和
if center > m * (10 / 8):
    标记为 hot pixel
```

4. 主图动态 hot/dead pixel 检测（算法在后面介绍）
5. PDAF line 标记， PDAF 异常点也写入同一个 `PixelsMap`，最后复用 DPC 插值。

显然这里不能检测大面积坏点簇
### 动态 hot/dead 检测算法

先在同 CFA 相位邻域上构造一个局部“正常值”估计，再看中心像素相对这个估计是否是局部离群点。它来自 Emil Martinec 的思路，代码里做了 OpenMP 和 SSE 优化。

第一步，对每个候选像素 `(i, j)`，在同色 3x3 采样网格上取中值（CFA上相同颜色）：

```text
median(
  raw[i-2][j-2], raw[i-2][j], raw[i-2][j+2],
  raw[i  ][j-2], raw[i  ][j], raw[i  ][j+2],
  raw[i+2][j-2], raw[i+2][j], raw[i+2][j+2]
)
```

第二步，计算中心像素相对中值的残差：

```text
cfablur[i][j] = rawData[i][j] - median_same_color_3x3(i, j)

pixdev > 0：中心比同色局部中值亮，候选 hot pixel
pixdev < 0：中心比同色局部中值暗，候选 dead pixel
```

第三步，用残差图的 5x5 局部能量作为自适应阈值参考：

```text
pixdev_abs = abs(cfablur[i][j])
hfnbrave = sum(abs(cfablur[5x5 around i,j])) - pixdev_abs
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

这个判据的含义是：中心残差必须相对邻域其它残差总和足够突出，才算坏点。不是固定亮度阈值。

边界处理上，边缘 2 像素不参与动态检测。

### Bayer 插值算法

“同色、成对、梯度加权”的插值。

每个候选方向都取一对位于坏点两侧的像素，如果这对像素中任意一个也被标记为坏点，该方向直接不用。

对绿色坏点，额外使用两个近对角方向：

```text
(row-1, col-1) + (row+1, col+1)
(row-1, col+1) + (row+1, col-1)

距离权重：
weight = 0.70710678 = 1 / sqrt(2)
```

对红/蓝坏点，对角同色点更远，使用：

```text
(row-2, col-2) + (row+2, col+2)
(row-2, col+2) + (row+2, col-2)

距离权重：
weight = 0.35355339 = 1 / sqrt(8)
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




## CFA line denoise

对raw图分通道做频域变换，在频域抑制周期性噪声。

**算法目的**
CFA line denoise 处理的是 RAW CFA 上沿行或列方向出现的固定模式噪声：

```text
horizontal line noise：横向条纹，整行或局部行段偏亮/偏暗
vertical line noise：纵向条纹，整列或局部列段偏亮/偏暗
PDAF line artifact：相位对焦行附近的横向异常
```

这类噪声如果留到 demosaic 后再处理，会被插值扩散到 RGB 三通道，变成更复杂的彩色条纹、假色或局部 banding。RawTherapee 把它放在 demosaic 前，在 CFA 原始采样结构中处理。

**算法流程**
```text
先分离低频/高频 （利用二维可分离的高斯滤波）
在每个 RGGB 子相位上做 8x8 DCT
只针对 DCT 中代表横线/竖线的轴向系数做 Wiener 风格衰减
再 inverse DCT 加回低频图
```
维纳滤波在这里有介绍 [[../deblur/基于PSF的去模糊|基于PSF的去模糊]] ，即：
```text
信号能量 coeffsq 大：noisefactor 接近 1，保留更多
噪声方差 noisevar 大：noisefactor 变小，衰减更多
```

高光保护和写回
```text
原始值接近 clipping：不改
处理后值接近 clipping：不写
```
这样做是为了避免在高光/接近饱和区域引入异常修正。高光区域本身非线性和 clipping 很强，DCT line denoise 的线性假设不稳。

## RAW CA correction


镜头横向色差表现为不同颜色通道在空间位置上不完全重合。典型现象：

```text
高反差边缘出现红/青边
高反差边缘出现蓝/黄边
画面边缘色边更明显
```

**总体算法结构**


1. 准备 CFA tile、缓存和临时绿色插值 Gtmp
   - 这里是经典操作，比较 R/G B/G 色差，先把G插全
   
2. 如果 autoCA：检测每个 tile 的 R/B 相对 G 的局部位移
   - 这里就是在局部区域内 shift R/B, 并计算色差 G-R G-B。这样得到很多不同shift的色差图。
   - 计算每个色差图的标准差，最小的就是最佳shift (颜色差异最小，意味着边缘处没有错位)
   - 高反差边缘权重大，因为 CA 最容易在边缘上估计；
   - 低频色差/颜色不稳定区域权重降低，避免把真实颜色边界当成 CA；
   - 亮度结构越强，估计越可信
   
   - 这里直接找出每个 tile 中的最小相差的位置
   
3. 把 tile shift 拟合成平滑的二维多项式位移场
   - 这里是全图做拟合，基于机头的色差是随半径变化的函数，不会突变。这样根据每个tilr的色差，拟合全图的分布，同时也能干掉离群点（拟合前也有滤波处理异常值）
   
4. 用 shift field 对 R/B 做重采样，并写回 rawData
   - 这里是根据 G + 前面拟合的分布 插值色差，然后进一步计算 R B 的值
   
5. 可选 avoidColourshift：用模糊的 per-pixel factor 抑制整体色偏
   - CA 重采样 R B 可能会造成这两个通道整体的颜色变了（低频变化）
   - 比较矫正前后 RB 亮度变化了多少，生成一个比例图
   - 对比例图做大半径高斯模糊（修正低频）
   - 再把比例图乘回矫正后的图像，实现抑制色偏

