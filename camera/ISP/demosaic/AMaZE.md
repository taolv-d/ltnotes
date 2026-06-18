---
type: note
status: done
tags:
  - camera
  - isp
rating: 0
create: 2026-06-01
update:
---

这里基于rawtherapee 实现介绍。由于代码非常复杂，这里只关注算法流程、思想，不深入细节。细节可以关注开源代码实现
链接：[[../rawtherapee|rawtherapee]]

AMaZE ( Aliasing Minimization and Zipper Elimination, 混叠最小化与拉链伪影消除) 算是 demosaic 中还原细节最好的算法，代价是算法复杂，反差区域可能出现伪彩。
AMaZE的核心假设有三个，这也是最新的 demosaic 算法的指导思想：
```text
1. green 通道采样最密，亮度结构主要由 green 承担，应优先高质量恢复 green。
2. G-R、G-B 色差在自然图像中比原始 R/G/B 更平滑，适合做方向选择和传播。
3. 普通边缘、Nyquist 高频纹理、对角边缘、高亮区域的插值风险不同，需要分情况处理。
```

源码里一张 tile 的主要流程可以概括成：

```text
tile 初始化和边界扩展
计算水平/垂直梯度权重
分别估计垂直/水平色差 vcd/hcd
根据色差方差和上下/左右插值波动得到 hvwt
检测 Nyquist 高频纹理并改用区域统计修正 hvwt
在 R/B 点恢复 green
用 green 曲率再次细化 Nyquist 区域
沿两条对角方向估计 R/B 关系，得到 pmwt 和 rbint
必要时用对角信息反向修正 green
插值 G-R、G-B 色差面
输出完整 RGB
```

上述流程看起来复杂，但本质就是计算两个权重（代码中hvwt）：
1. 插值出完整的 G 通道（以R通道插值为例），核心就是插值权重
	- 利用插值点左右、上下的 G-G 计算梯度，利用 R-R 计算上下/左右的差异，以此确定初步的插值的权重
	- 针对高频纹理区域（Nyquist 区域），修正插值权重（倾向于平滑区域）
	- 利用 R B 斜对角分布的特性，拿到斜向信息，进一步修正 插值权重
2. 插值出 R B 色差通道，核心也是插值权重
	- R B 通道是斜向分布的，这里估计的权重也是斜向搞

整个算法流程中，计算插值权重。算法实现中比较细碎，但整体来看每个插值点都考虑了 上下左右+4个斜向 共8个方向：
- 对于G 通道，第一步先计算上下左右的权重，在 R B 插值结束后 又利用 R B 的斜向信息进行一轮修正
- 对于 R B，他们的色差（R-B 之间）是通过斜向的关系计算的。最后插值G通道的色差时利用了上下左右的信息

本算法中计算色差、估计权重过程重复出现了如下算法结构：
```
1.色差法+比例法估计色差
2.利用色差平滑+结构信息得到最终的色差
3.基于两个方向的梯度（平滑程度），确定两个方向的权重（色差判据+结构判据）
4.溢出处理
```
着套结构在初步进行G通道插值、估计斜向R-B色差，根据R-B修正G插值 都有使用。
这套算法结构中出现两次抉择：1是两种算法计算的色差该信那个，判据利用了色差缓慢变化的特性，优先选择变化小的。2是两个梯度判据（色差判据+结构判据）应该用那个

### 第一步：计算水平/垂直梯度权重

AMaZE 先在原始 CFA 上估计局部方向强度：

```cpp
// 当前点的梯度值
delh = abs(cfa[x+1] - cfa[x-1])
delv = abs(cfa[y+1] - cfa[y-1])

// 当前点、当前点两侧的梯度变化是不是很大，越大说明有纹理，不适合插值
// dirwts0 是垂直方向的不平滑程度
dirwts0 = eps
        + abs(cfa[y+2] - cfa[x])
        + abs(cfa[x] - cfa[y-2])
        + delv

// dirwts1 是水平方向的不平滑程度
dirwts1 = eps
        + abs(cfa[x+2] - cfa[x])
        + abs(cfa[x] - cfa[x-2])
        + delh

delhvsqsum = delh^2 + delv^2
```

### 第二步：估计 R B 位置的色差（G-R, G-B）

这里又分为四个子步骤：
1. 分别使用色差法、比例法初步估计
2. 平滑色差估计
3. 融合水平垂直色差
4. 恢复G
这里就是前面提到的重复出现的算法结构的具体实现的例子了。后续也有类似的结构，虽然有变换但，整体思想是一致的
#### 1. 色差法、比例法初步估计色差
对每个待估计位置，AMaZE 同时计算四个方向的 green 候选值：上、下、左、右。每个方向都有两种估计：

```text
1. 色差法（Hamilton-Adams）：加法模型
2. 比例法（adaptive ratio）：乘法模型
然后在两者之间选择更可信的那个。
```

以向上方向为例，**Hamilton-Adams 形式是**：

```text
guha = G_up1 + 0.5 * (R_current - R_up2)

这正是 Hamilton-Adams 在一个方向上的分解：
G_est = G_known + 梯度修正
梯度 ≈ (当前同色值 - 该方向上另一个同色值) / 2
除以 2 是因为从 up2 到 current 跨越了两个单位距离（中间隔了一个点）。
```

**adaptive ratio 则先估计颜色比例**：
核心就是这个公式，但是这里加上了平滑权重
```Cpp
G_est = R_current × (G_known / R_known) //简化公式

cru = cfa[up1] * (dirwts0[up2] + dirwts0[x])
    / (dirwts0[up2] * (eps + cfa[x])
       + dirwts0[x] * (eps + cfa[up2]))
```

如果比例接近 1，说明局部颜色比例比较可信，比例偏离太大时不用 ratio，退回 Hamilton-Adams，避免被异常颜色比例或噪声带偏。在纯色区域（某一项接近0时比例法失效）

四个方向都得到候选后，AMaZE 用相邻方向的梯度权重做上下/左右合成：

```text
vwt = dirwts0[up1] / (dirwts0[down1] + dirwts0[up1])
hwt = dirwts1[left1] / (dirwts1[left1] + dirwts1[right1])

Gintv = vwt * G_down + (1 - vwt) * G_up
Ginth = hwt * G_right + (1 - hwt) * G_left
```

这里的权重写法有一个直观解释：如果 down 方向梯度大，分母变大，`vwt` 变小，结果更偏向 up；如果 up 方向梯度大，`vwt` 变大，结果更偏向 down。最终偏向较平滑的一侧。

然后保存两套色差：
- 在平滑区域，adaptive ratio 可能更好（颜色比例恒定）
- 在边缘或高对比区域，adaptive ratio 可能被梯度干扰而失真，Hamilton-Adams 更稳定

#### 2. 平滑假设，得到最终水平/垂直色差

第二步准备了两套色差，最终应该用谁呢？色差理论指出：**色差面应该平滑**。因此这里比较各自方法在水平/竖直方向连续估计的3个色差值得方差，那个小，那个有效。

```text
hcdvar    = Var(hcd[x-2],    hcd[x],    hcd[x+2])
hcdaltvar = Var(hcdalt[x-2], hcdalt[x], hcdalt[x+2])

如果 hcdaltvar 更小，则用 hcdalt 替换 hcd。

vcd 同理，比较 vcd[y-2], vcd[y], vcd[y+2]。
```

高亮区域还有额外限制。当原始值或插值 green 接近裁剪时，颜色比例会失真：

```text
1. 接近 clip_pt8 时优先使用 Hamilton-Adams。
2. 插值结果超过 clip_pt 或色差符号/幅度不合理时，用邻域 median 约束。
```

这一步可以理解成高亮保护：避免在饱和边缘附近产生彩边、色块和负值。

#### 3. 计算水平/垂直的融合权重（hvwt）

`hvwt` 是 AMaZE 恢复 green 的核心。它不是只看原始梯度，而是结合两类证据：

```text
1. 色差面方差：哪个方向的 G-R/G-B 更平滑。
2. 对向插值波动：上/下估计是否一致，左/右估计是否一致。
```

**色差方差**：
方差统计也是沿着水平/垂直方向向外延申，计算这些色差值的方差
色差方差大，说明该方向不可靠，应提高另一个方向的权重（权重可以用方差的大小反应）

**对向插值波动**
如果上/下两个估计差异大，垂直方向不稳定；如果左/右两个估计差异大，水平方向不稳定。显然也是稳定的方向会分配更高的权重


**接下来的问题是两个方法给出的结果该信谁？**

1. **如果两位参谋意见一致**：
    - 说明垂直/水平方向的优劣非常明确。
    - 此时，算法会选择**意见更“强烈”**的那一位。所谓“强烈”，就是它的值更偏离 0.5（比如 0.9 或 0.1）。

2. **如果两位参谋意见不一致**（一个说偏垂直，一个说偏水平）：
    - 这说明情况很复杂，局部纹理可能很难判断。
    - 此时，算法会采用**更保守的建议**，即听从 **对向插值波动**。因为在纹理复杂、颜色容易出错的地方，“结构稳定性”（上下/左右是否一致）通常比“颜色平滑度”更可靠。

#### 4. 在 R/B 点恢复 green

有了 `vcd/hcd/hvwt` 后，AMaZE 开始真正写入 R/B 点的 green。
先看对角邻居的 `hvwt` 是否给出更强方向判断：

```text
hvwtalt = (hvwt[NW] + hvwt[NE] + hvwt[SW] + hvwt[SE]) / 4

如果 hvwtalt 比当前 hvwt 更远离 0.5，
说明邻域方向判断更明确，则用 hvwtalt 替换当前 hvwt。
```

然后融合水平/垂直色差：

```text
Dgrb = hvwt * vcd + (1 - hvwt) * hcd
green = cfa + Dgrb
```

### 第三步：Nyquist 处理

这里分为三个小步骤：
1. Nyquist 高频纹理检测
2. Nyquist 区域初步插值
3. 用 green 曲率细化 Nyquist 区域
#### Nyquist 高频纹理检测

高频纹理的判据： **水平和垂直色差估计差异很大**，如果水平/垂直色差估计互相矛盾，并且这种矛盾相对局部梯度足够大，就认为该区域可能是 Nyquist 高频纹理。

Nyquist **初始区域**计算方法如下：

```text
nyqutest =
    Gaussian_odd_5x5(cddiffsq)
  - Gaussian_grad_5x5(delhvsqsum) * nyqthresh

nyqutest > 0, 则为 Nyquist 区域

其中：
cddiffsq = (vcd - hcd)^2
delhvsqsum = 水平/垂直 CFA 梯度平方和
nyqthresh = 0.5

这里的逻辑是：色差的梯度跟CFA原始图上的梯度强度不会有巨大差异，如果插值出来的色差梯度很大，超过原有CFA中的梯度强度，那就说明插值出来的色差有问题，大概率出现假色
```

接下来，用 8 邻域投票做一次**形态学修正**（滤掉孤立点）：

```text
8 个同 coset 邻居中：
超过 4 个是 Nyquist -> 当前点置 1
少于 4 个是 Nyquist -> 当前点置 0
正好 4 个 -> 保持原状态
```

这一步生成 `nyquist2`，作用是让 Nyquist 区域更连贯，同时过滤孤立误检点。

#### Nyquist 区域初步插值

对 `nyquist2=1` 的区域，AMaZE 做区域统计（相当于低通了）。它在当前点周围 `[-6, 6]`、步长 2 的同色采样网格上，只统计同样被标为 Nyquist 的点。

同样还是计算水平、竖直方向的方差，然后计算水平竖直的权重。

#### 用 green 曲率细化 Nyquist 区域

对 Nyquist 点，需要计算 green 差分（实际也是评估那个方向更平滑）：

```text
Dgrb2.h = (green[x] - 0.5 * (green[left] + green[right]))^2
Dgrb2.v = (green[x] - 0.5 * (green[up]   + green[down]))^2
```

AMaZE 会对 `nyquist2=1` 的点再次修正 green，这里也是修正权重：

```text
gvarh = epssq + weighted_average(Dgrb2.h)
gvarv = epssq + weighted_average(Dgrb2.v)
```

然后重新融合色差（平滑的方向给更高的权重，这样抑制伪色）：

```text
Dgrb = (hcd * gvarv + vcd * gvarh) / (gvarv + gvarh)
green = cfa + Dgrb
```

这个式子同样是“用对方的方差做权重”的形式：

```text
水平 green 差异大 -> 水平方向不稳定 -> 提高垂直 vcd 权重。
垂直 green 差异大 -> 垂直方向不稳定 -> 提高水平 hcd 权重。
```

### 第四步：对角方向修正G

#### 1.估计 R/B 关系（插值 R-B B-R）

在对G通道插值时, 实际只用到了上下左右的像素，对于呈斜对角分布的R、B，插值的时候没有用到。这里进一步插值出 R B之间的色差，用于反过来修正G插值。

两个方向得插值结果也是利用方差做权重，相信变化更平缓的一侧。这里会得到权重表 pmwt

这里的算法跟 第一步、第二步 这里插值G的算法完全一致, 只不过插值的目标不是G

#### 2. 用对角信息反向修正 green

第二步插值G时只用到了水平、竖直方向的信息，没有考虑到斜对角方向。现在有了斜对角方向 R-B 色差信息，可以用于修正 G 通道的插值（例如斜向纹理）。

如果**对角线的方向判断不比正交方向弱**，说明对角方向能提供正交方向没有的信息，值得用来修正。
判断条件就是权重：
```text
如果 abs(0.5 - pmwt) < abs(0.5 - hvwt)，跳过。
否则说明对角方向的判别强度不弱于水平/垂直判别，尝试修正 green。
```

这里修正得意思是：用 R+B 的亮度/结构信息来指导G的插值（这里不是完整的亮度信息），也是假设在局部区域内R + B是缓慢变化的。利用R+B的变化来看G插值是否符合这个变化规律，有差异则修正。

修正过程也是前面说的算法结构：
```text
1. 用 rbint 构造上、下、左、右四个方向的 green 候选。
2. 每个方向仍然在 adaptive ratio 和 HA-like 形式之间选择。
3. 用 dirwts0/dirwts1 合成 Gintv/Ginth。
4. 做 median/clip 约束。
5. 用 hvwt 融合：

   green = hvwt * Gintv + (1 - hvwt) * Ginth
   Dgrb  = green - cfa
```

### 第五步：计算G-R / G-B 色差面

到目前为止，已经估计出来的色差信息有：

|当前位置|原始 CFA|Dgrb[0] (G-R)|Dgrb[1] (G-B)|
|---|---|---|---|
|R 点|R|= Dgrb（已知）|缺失|
|B 点|B|缺失|= Dgrb（已知）|
|G 点|G|缺失|缺失|

对于每个需要插值的点，在它的四个斜向进行插值（权重固定）（以填补 G-R 为例）。对于某个缺失 G-R 的位置，考虑四个对角方向：NW、NE、SW、SE。

**每个方向的权重计算**：

```cpp
wtnw = 1 / (eps + 该方向上的色差变化程度) // 色差就是G-R这种
```

色差变化越小 → 权重越大 → 该方向更可信

**每个方向的估计值**（不是简单取最近邻，而是用外推组合）：

```cpp

estimate_nw = 1.325 × Dgrb[NW1] 
            - 0.175 × Dgrb[NW3]
            - 0.075 × Dgrb[NW1-left2]
            - 0.075 × Dgrb[NW1-up2]
```

|系数|含义|
|---|---|
|1.325|主权重，来自最近的有效点|
|-0.175|负权重，补偿次近点|
|-0.075 × 2|负权重，补偿侧向点|
相当于一个**带方向性的 FIR 低通滤波器**，在平滑色差的同时保留对角线方向的结构。

最终：
```text
Dgrb[c] =
    (wtnw * estimate_nw
   + wtne * estimate_ne
   + wtsw * estimate_sw
   + wtse * estimate_se)
    / (wtnw + wtne + wtsw + wtse)
```

这一段的思想是：色差面应该平滑，但边缘方向上更可靠，所以用“方向一致性越强，权重越大”的对角插值恢复缺失 chroma。

### 第六步：输出 RGB
这一步已经有了：
- 完整的绿色通道，缺失部分已经插值+修正
- R、B通道的色差，根据G通道插值可以计算

这里还缺：G 通道的色差，且他的上下左右都已经估计出色差了。因此直接用前面计算的梯度权重`hvwt`进行加权计算即可。

```text
色差 = (
    hvwt[up]       * Dgrb[up]
  + (1-hvwt[right]) * Dgrb[right]
  + (1-hvwt[left])  * Dgrb[left]
  + hvwt[down]     * Dgrb[down]
) / normalization
```

现在色差、权重都已经有了，可以直接恢复完整的R G B 三通道

```text
red   = max(0, 65535 * (green - Dgrb[0]))
blue  = max(0, 65535 * (green - Dgrb[1]))
green = max(0, 65535 * rgbgreen)
```


