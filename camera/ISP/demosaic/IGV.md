
定位：

IGV 全称是 Integrated Gaussian Vector on Color Differences。它不像 AMaZE 那么复杂，也不像 RCD 那样把方向图 `VH_Dir / PQ_Dir` 显式写出来，但大框架并没有本质区别，仍然是：

```text
1. 先在 R/B 点恢复 green
2. 把色差 G-R / G-B 作为主要传播对象
3. 根据局部色差稳定性给方向权重
4. 沿更平滑方向传播 chroma
```

一句话概括它的实现逻辑：

```text
先在 R/B 点估计水平/垂直 G-C 色差，
再用 integrated gaussian vector 形式评价哪个方向更稳定，
恢复完整 green，
然后在色差面上补 R@B / B@R / R@G / B@G。
```

这里 `C` 表示当前位置真实采样的颜色，可能是 R，也可能是 B。

### 7.1 buffer 设计

IGV 主要维护四类中间量：

```text
rgb[0] = R
rgb[1] = G
rgb[2] = B

vdif   = 垂直方向估计的 G-C 色差
hdif   = 水平方向估计的 G-C 色差

chr[0] = G-R
chr[1] = G-B
```

输出时直接使用：

```text
R = G - (G-R)
B = G - (G-B)
```

源码最后就是：

```cpp
red   = green - 65535 * chr[0]
blue  = green - 65535 * chr[1]
```

所以 `chr` 是算法真正维护的 chroma 差分面。

### 7.2 Step 1：把 CFA 放到对应颜色平面

源码先把 rawData 写入 CFA 对应的颜色平面：

```cpp
int c = FC(row, col);
rgb[c][indx] = rawData[row][col];
```

这时：

```text
R 点只有 R
G 点只有 G
B 点只有 B
```

接下来的目标是：

```text
先补 green
再补 chr[0]=G-R 和 chr[1]=G-B
```

### 7.3 Step 2：在 R/B 点生成 vdif / hdif

对每个 R/B 点，IGV 先构造 N/E/W/S 四个方向梯度：

```text
ng, eg, wg, sg
```

例如北方向：

```text
ng = eps
   + abs(G[N1] - G[N3])
   + abs(C[x]  - C[N2])
```

含义：

```text
如果北方向 green 变化大，或者同色 C 变化大，
说明北方向不平滑，不适合沿北方向外推 green。
```

然后在四个方向上分别做高阶 green 候选估计：

```text
nv = north green estimate
ev = east  green estimate
wv = west  green estimate
sv = south green estimate
```

北方向公式是：

```cpp
nv =
(
  23 * G[N1]
+ 23 * G[N3]
+      G[N5]
+      G[S1]
+ 40 * C[x]
- 32 * C[N2]
-  8 * C[N4]
) / (48 * 65535)
```

可以理解成：

```text
用北侧多级 green 采样做主支撑，
再用当前 R/B 样本和更远同色样本做高阶校正。
```

它不是最终 green，而是“方向候选 green”。比简单的 Hamilton-Adams：

```text
G ~ G[N1] + 0.5 * (C[x] - C[N2])
```

支撑更宽，目的是：

```text
降低噪声
减少锯齿
减少 maze / posterization
```

然后先做正交方向融合，再转成色差：

```cpp
vdif = (sg * nv + ng * sv) / (ng + sg) - C[x]
hdif = (wg * ev + eg * wv) / (eg + wg) - C[x]
```

这里依然是“对侧梯度加权”：

```text
ng 大 -> north 不可靠 -> vdif 更偏 south 候选
sg 大 -> south 不可靠 -> vdif 更偏 north 候选
```

所以到这一步，IGV 得到的是：

```text
vdif = 垂直方向 G-C 色差
hdif = 水平方向 G-C 色差
```

### 7.4 Step 3：IGV 核心，融合 vdif / hdif

这是 IGV 名字里的 “Integrated Gaussian Vector” 主要体现的地方。

源码对 `vdif` 和 `hdif` 分别构造一个长公式：

```text
ng = vertical integrated variance-like measure on vdif
eg = horizontal integrated variance-like measure on hdif
```

代码注释是：

```text
H&V integrated gaussian vector over variance on color differences
```

可以把它理解成：

```text
ng 衡量垂直色差场的局部不稳定程度
eg 衡量水平色差场的局部不稳定程度
```

它不是简单的 3 点方差，而是对 `±2, ±4, ±6` 范围内的色差平方项和局部组合项做积分型加权。设计目标是：

```text
不只看单点方向冲突，
而是看这个方向上的色差场整体是否稳定。
```

在真正融合前，IGV 还会先对 `vdif` / `hdif` 做一次受 median 约束的局部平滑：

```cpp
nv = median(
    0.725  * vdif[x]
  + 0.1375 * vdif[up2]
  + 0.1375 * vdif[down2],
    vdif[up2],
    vdif[down2]
)

ev = median(
    0.725  * hdif[x]
  + 0.1375 * hdif[left2]
  + 0.1375 * hdif[right2],
    hdif[left2],
    hdif[right2]
)
```

这一步的作用：

```text
轻微平滑色差
同时用 median 限制异常值
```

最后融合：

```cpp
chr[d][indx] = (eg * nv + ng * ev) / (ng + eg);
```

这里和 RCD、AMaZE 的一些权重写法一样，也是“用对方的不稳定度做权重”：

```text
水平色差不稳定 eg 大 -> 更偏垂直色差 nv
垂直色差不稳定 ng 大 -> 更偏水平色差 ev
```

然后恢复 green：

```cpp
rgb[1][indx] = rgb[c][indx] + 65535 * chr[d][indx];
```

也就是：

```text
G = C + (G-C)
```

到这一步，R/B 点上的 green 就补完整了。

### 7.5 Step 4：在 R/B 点补另一个颜色

完成 green 后：

```text
R 点还缺 B
B 点还缺 R
```

IGV 在 `chr` 色差面上用四个对角方向传播 chroma。

先构造四个对角方向权重：

```text
nwg, neg, swg, seg
```

形式都类似：

```text
weight = 1 / (eps + 该对角方向上的色差变化)
```

色差变化越小，权重越大。

然后对每个方向的候选色差做 median 限制：

```cpp
nwv = median(chr[NW1], chr[NW3-row], chr[NW3-col])
nev = median(chr[NE1], chr[NE3-row], chr[NE3-col])
swv = median(chr[SW1], chr[SW3-row], chr[SW3-col])
sev = median(chr[SE1], chr[SE3-row], chr[SE3-col])
```

最后：

```cpp
chr[c][indx] =
    (nwg * nwv + neg * nev + swg * swv + seg * sev)
  / (nwg + neg + swg + seg);
```

本质是：

```text
沿更平滑的对角方向传播 G-R / G-B 色差，
补出 R@B 或 B@R。
```

源码分两个 row loop 处理不同 Bayer parity，但算法逻辑相同。

### 7.6 Step 5：在 G 点补 R/B 色差

在 G 点上，IGV 要补：

```text
chr[0] = G-R
chr[1] = G-B
```

它分别对 `chr[0]` 和 `chr[1]` 做一次 N/E/W/S 加权传播。

权重：

```cpp
ng = 1 / (eps + vertical chroma variation)
eg = 1 / (eps + east chroma variation)
wg = 1 / (eps + west chroma variation)
sg = 1 / (eps + south chroma variation)
```

然后：

```cpp
chr[c][indx] =
    (ng * chr[c][N]
   + eg * chr[c][E]
   + wg * chr[c][W]
   + sg * chr[c][S])
  / (ng + eg + wg + sg);
```

含义和前面一致：

```text
哪个方向的色差面更连续，就更相信哪个方向。
```

### 7.7 Step 6：输出 RGB

最后统一输出：

```cpp
red   = green - 65535 * chr[0]
green = green
blue  = green - 65535 * chr[1]
```

边界调用：

```cpp
border_interpolate(winw, winh, 8, rawData, red, green, blue)
```

### 7.8 IGV 精简伪代码

```text
copy raw CFA samples into sparse rgb planes

for each R/B site:
    compute N/E/W/S gradients
    estimate N/E/W/S green by high-order directional interpolation
    fuse north/south -> vdif = G-C
    fuse west/east  -> hdif = G-C

for each R/B site:
    compute integrated variance-like stability for vdif and hdif
    median-limit local vdif/hdif
    fuse vertical and horizontal colour difference
    recover green = C + (G-C)

for each R/B site:
    compute diagonal chroma weights
    median-limit diagonal chroma candidates
    interpolate chr on diagonals

for each G site:
    compute N/E/W/S chroma weights
    interpolate chr horizontally/vertically

recover:
    R = G - (G-R)
    B = G - (G-B)

border_interpolate(...)
```

### 7.9 和 RCD/LMMSE/AMaZE 的关系

IGV 和它们在大框架上没有本质差别，仍然属于：

```text
先 green
再色差
方向加权
最后输出 RGB
```

差异主要是工程实现和权重设计：

```text
AMaZE:
    最复杂，Nyquist、rbint、fancy chroma 都有。

RCD:
    VH/PQ 方向图清晰，ratio correction 工程感强。

LMMSE:
    用局部方差/残差方差做 MMSE 融合，更偏噪声稳定。

IGV:
    用 vdif/hdif 的 integrated variance 和局部 chroma 连续性决定方向权重。
    作为高 ISO / fallback 算法很实用，但质量上限通常低于 AMaZE/RCD。
```

工程评价：

```text
优点：
    主线清楚。
    全程围绕色差面传播，逻辑统一。
    对噪声和假色比简单方向插值更稳。

局限：
    没有 AMaZE 那样复杂的特殊场景保护。
    没有 RCD 那样显式、可读性很强的方向图结构。
    integrated gaussian variance 公式偏经验化，代码可读性不如 RCD。
```