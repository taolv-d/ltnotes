
AWB 算法就是找到图像中灰色的部分，以此为依据，调整三个颜色通道的增益。


## Grey World

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


## White Patch

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

## Shades of Grey

Shades of Grey 是 Grey World 和 White Patch 的统一形式，使用 Minkowski 范数：

$$
L_p(C) = {(1/N * \sum_i {C_i}^p)}^{(1/p)}
$$

当 `p=1` 时接近 Grey World；当 `p` 趋近无穷大时接近 White Patch。

增益：

```text
gain_C = K / L_p(C)
```


## 饱和像素剔除

饱和像素不再反映真实通道比例，需要从统计中剔除。常见规则：

```text
if R > T or G > T or B > T:
  ignore pixel
```

或只剔除接近满量程的通道。AWB 文档中有 `remove_en`、`remove_T` 之类概念，其目的就是降低高光 clipping 对光源估计的污染。


## 工程注意事项

- AWB 统计最好在 LSC 后进行，否则暗角和色彩阴影会污染平均值。
- 大面积单色场景会违反 Grey World 假设，例如草地、蓝天、红墙。
- 高亮区域不一定是白色，White Patch 在舞台灯、彩灯场景容易失效。
- AWB 输出增益应做限幅和平滑，否则视频中会出现色温跳变。