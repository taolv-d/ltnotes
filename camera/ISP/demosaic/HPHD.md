

```text
1. 先用高通响应构造方向图 hpmap
2. 在 R/B 点按 hpmap 决定 green 用水平、垂直还是四方向混合插值
3. green 恢复完成后，再用一个比较基础的 C-G 色差插值补 red/blue
```

一句话：

```text
HPHD 的重点几乎都在 green 恢复；
red/blue 补全部分相对简单。
```

#### HPHD 的 `hpmap`

源码先做两步：

```text
hphd_vertical(...)
hphd_horizontal(...)
```

先在垂直方向上计算一个宽支撑高通响应：

```text
temp = abs(
    (x[-5] - x[+5])
  - 8  * (x[-4] - x[+4])
  + 27 * (x[-3] - x[+3])
  - 48 * (x[-2] - x[+2])
  + 42 * (x[-1] - x[+1])
)
```

然后在 9 点窗口里求：

```text
avg = 局部平均高通响应
dev = 局部高通波动
```

再得到垂直方向的平滑高通图：

```text
hpmap_vertical = avgL + (avgR - avgL) * devL / (devL + devR)
```

接着 `hphd_horizontal()` 再算水平方向的对应高通量 `hpv`，并和前面的值比较，最后把 `hpmap` 量化成三类：

```text
hpmap == 1:
    水平方向优先

hpmap == 2:
    垂直方向优先

hpmap == 0:
    水平/垂直接近，四方向混合
```

判断条件是：

```text
如果 vertical 明显强于 horizontal -> 选水平插值
如果 horizontal 明显强于 vertical -> 选垂直插值
否则 -> 两者混合
```

也就是：

```text
哪个方向高通结构更强，就更不应该沿那个方向插 green。
```

#### green 怎么插

真正恢复 green 在：

```text
hphd_green(...)
```

如果 `hpmap == 1`，只做水平方向插值。构造左右两个候选：

```text
g2 = G[right1] - 0.5 * C[right2]
g4 = G[left1]  - 0.5 * C[left2]
```

再算两个方向的权重：

```text
e2 = 1 / (局部差分和)
e4 = 1 / (局部差分和)
```

最后：

```text
G = 0.5 * C[x] + (e2 * g2 + e4 * g4) / (e2 + e4)
```

如果 `hpmap == 2`，同理改成垂直方向：

```text
g1 = G[up1]   - 0.5 * C[up2]
g3 = G[down1] - 0.5 * C[down2]

G = 0.5 * C[x] + (e1 * g1 + e3 * g3) / (e1 + e3)
```

如果 `hpmap == 0`，四个方向都参与：

```text
g1, g2, g3, g4
e1, e2, e3, e4

G = 0.5 * C[x] + weighted_average(g1, g2, g3, g4)
```

这里的 `e1..e4` 都是反差分权重：

```text
局部变化越小 -> 权重越大
局部变化越大 -> 权重越小
```

所以 HPHD 的 green 恢复可以概括成：

```text
高通方向图先决定“走横向、纵向还是混合”，
然后在被选中的方向上做局部平滑性加权。
```

#### red/blue 怎么补

这部分比 green 简单很多。HPHD 直接调用：

```cpp
interpolate_row_rb_mul_pp(...)
```

实现在：

```text
rawimagesource_i.h
```

它不是直接对四周 raw 值做平均，而是以已经恢复好的 green 为基准，做一个基础的 `C-G` 色差插值。

在 R 点补 B，或在 B 点补 R 时：

```text
用四个对角邻居
平均的是 (邻居颜色 - 邻居 green)
最后再加回当前 green
```

也就是：

```text
C_missing = G_current + average(C_diag - G_diag)
```

在 G 点补 R 或 B 时：

```text
水平或垂直做线性 C-G 插值

C@G = G_current + 0.5 * ((C-G)left + (C-G)right)
或
C@G = G_current + 0.5 * ((C-G)up + (C-G)down)
```

所以红蓝补全的特点是：

```text
不是直接平均四周 raw
而是简单的色差插值
但这部分没有 AMaZE/RCD 那样复杂的方向判别和保护
```

#### 怎么理解 HPHD

它更像：

```text
高通边缘检测
-> 方向分类
-> 定向插 green
-> 常规色差法补 red/blue
```

和几种更现代的 Bayer 算法相比：

```text
优点：
    green 恢复逻辑清楚
    高通图驱动方向选择，工程实现不算难

局限：
    red/blue 补全部分偏简单
    没有 AHD 的 Lab 同质性选择
    没有 RCD 的显式方向图 + ratio correction
    没有 AMaZE 的 Nyquist / 对角修正 / fancy chroma
```
