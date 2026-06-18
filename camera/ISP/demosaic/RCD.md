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

RCD 全名是 Ratio Corrected Demosaicing，他比 [[AMaZE]] 算法上要简洁的多。
### RCD 和 AMaZE 的核心差异

**方向判断不同**
1. AMaZE 的 `hvwt` 来自多类证据：色差方差，上下/左右插值一致性，Nyquist 区域特殊处理，斜向R B的补充等。把能利用的信息都用了，但是代价就是算法复杂

2. RCD 的方向判断更直接。它先计算高通结构响应，主要基于水平、竖直两个方向计算权重

**green 插值不同**
1. AMaZE 先用`ratio + Hamilton-Adams` 两种方法得到水平垂直的色差；再通过 水平垂直权重 融合；还会进行Nyquist区域修正；斜向修正。

2. RCD 插 green 更直接：
```text
先构造 lpf
用 lpf 做 ratio correction
得到 N/S/W/E 四个方向估计
用 VH_Dir 融合 V_Est/H_Est
```

**高频处理不同**
AMaZE 显式检测 Nyquist 高频纹理：
RCD 没有

**红蓝恢复不同**

AMaZE 会：沿对角估计色差，同时会反向修正G

RCD 更直接：
```text
在 R/B 点沿对角方向插 R-G 或 B-G 色差
在 G 点沿水平/垂直方向插 R-G / B-G 色差
```

### Step 1：建立水平/垂直方向图 VH_Dir

对比AMaZE，这里也是计算水平垂直梯度的权重，但简单的多。这里的梯度权重不用于G通道的差值，再后续R B 通道的差值会用到

RCD 先计算垂直和水平的高通响应平方。垂直方向计算方法如下（水平方向同理，不写了）：
这里使用了这样的滤波器找高频： [1, -3, -1, 6, -1, -3, 1]
```text
V_HPF =
    (cfa[y-3] - cfa[y-1] - cfa[y+1] + cfa[y+3])
  - 3 * (cfa[y-2] + cfa[y+2])
  + 6 * cfa[x]

V_response = V_HPF^2
```

然后在小邻域内累计：
```text
V_Stat = max(epssq, V0 + V1 + V2)
H_Stat = max(epssq, H0 + H1 + H2)

VH_Dir = V_Stat / (V_Stat + H_Stat)
```

直观含义：
```text
V_Stat 大：垂直方向变化强，垂直插值风险更高，应更偏水平。
H_Stat 大：水平方向变化强，水平插值风险更高，应更偏垂直。
```

### Step 2：G 差值

#### 构造 low-pass filter

RCD 先为同色采样网格构造一个低通量 `lpf`：

```text
lpf =
    cfa[x]
  + 0.5 * (N + S + W + E)
  + 0.25 * (NW + NE + SW + SE)
```

它混合了当前 CFA、上下左右和四个对角邻域。作用是给 ratio correction 提供一个更平滑的**局部结构参考**。

#### 在 R/B 点插 green

这是 RCD 的核心 green 恢复步骤。

先计算四个 上下左右四个方向的梯度， 也是判断这个方向是否平滑（色差越平滑越越可靠）：
```text
N_Grad, S_Grad, W_Grad, E_Grad
```

例如北方向：
```text
N_Grad = eps
  + abs(cfa[N] - cfa[S])
  + abs(cfa[x] - cfa[N2])
  + abs(cfa[N] - cfa[N3])
  + abs(cfa[N2] - cfa[N4])
```

然后用 `lpf` 做 ratio-corrected green 估计。以北方向为例：
```text
N_Est = cfa[N] * (2 * lpf[x]) / (eps + lpf[x] + lpf[N])

含义：
cfa[N] 是相邻 green 采样值。
lpf[x] / lpf[N] 描述当前点和邻点的低频亮度/结构比例。
用这个比例去修正直接拿邻居 green 的结果。
```

四个方向得到：
```text
N_Est, S_Est, W_Est, E_Est
```

再合成垂直/水平候选：
```text
V_Est = (S_Grad * N_Est + N_Grad * S_Est) / (N_Grad + S_Grad)
H_Est = (W_Grad * E_Est + E_Grad * W_Est) / (E_Grad + W_Grad)
```

这里仍然是“用对侧梯度做权重”的形式：
```text
北方向梯度大 -> N_Est 不可靠 -> V_Est 更偏 S_Est
南方向梯度大 -> S_Est 不可靠 -> V_Est 更偏 N_Est
```

最后根据色差+权重在 R 或 B 采样点上恢复 green。

### Step 3：R B 恢复另一个颜色
#### 建立对角方向图 PQ_Dir

red/blue 在 Bayer 中处于对角采样关系，所以 RCD 恢复 R/B 时使用两条对角方向

形式和 **建立水平/垂直方向图 VH_Dir** 的高通结构相似，只是方向换成对角线

#### 在 R/B 点恢复另一个颜色
此时G已经完成插值，需要**在 R 点要恢复 B，在 B 点要恢复 R**。这里也是沿斜向插色差：
这里算法也是经典的思想：
1. 沿四个对角方向的色差候选
2. 构造四个对角梯度
3. 根据梯度加权，平滑的一边权重更大

权重这里关注：
  1. 当前对角线两端目标颜色 C 的差异
  2. 更远处同方向 C 的连续性
  3. 当前 green 和远处 green 的结构差异

插值这里实际是分两步走的：
1. 先有四个斜向+这里计算的权重 得到两个对角线的权重
2. 两个对角线合成，使用上一步计算的斜向权重再融合得到最终的结果
```
  NW_Est ----\
              -> P_Est ----\
  SE_Est ----/              \
                              -> final C-G -> C = G + final C-G
  NE_Est ----\              /
              -> Q_Est ----/
  SW_Est ----/
```
### Step 4：在 green 点恢复 R/B

到这一步就只剩G通道对应色差还没有

在 green 采样点上，RCD 要同时恢复 R 和 B。此时使用水平/垂直方向的色差传播。

先重新取 `VH_Disc`：

```text
VH_Disc = 中心 VH_Dir 和邻域 VH_Dir 中方向性更强者
```

然后对 R、B 两个通道分别做同样处理：

```text
for C in {R, B}:
    N_Est = C[N] - G[N]
    S_Est = C[S] - G[S]
    W_Est = C[W] - G[W]
    E_Est = C[E] - G[E]
	//四个方向根据梯度加权合并成两个方向
    V_Est = (N_Grad * S_Est + S_Grad * N_Est) / (N_Grad + S_Grad)
    H_Est = (E_Grad * W_Est + W_Grad * E_Est) / (E_Grad + W_Grad)
	// H V 两个方向再根据第一步的权重合并 
    C@G = G + intp(VH_Disc, H_Est, V_Est)
```


