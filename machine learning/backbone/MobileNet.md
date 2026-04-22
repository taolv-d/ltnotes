# V1
假设输入是 224x224x3，width_multiplier=1：

|层名|操作|输出尺寸|
|---|---|---|
|conv_1|3x3 conv, stride=2, 32ch|112x112x32|
|conv_ds_2|DW 3x3 + PW 1x1, 64ch|112x112x64|
|conv_ds_3|DW stride=2 + PW, 128ch|56x56x128|
|conv_ds_4|DW + PW, 128ch|56x56x128|
|conv_ds_5|DW stride=2 + PW, 256ch|28x28x256|
|conv_ds_6|DW + PW, 256ch|28x28x256|
|conv_ds_7|DW stride=2 + PW, 512ch|14x14x512|
|conv_ds_8|DW + PW, 512ch|14x14x512|
|conv_ds_9|DW + PW, 512ch|14x14x512|
|conv_ds_10|DW + PW, 512ch|14x14x512|
|conv_ds_11|DW + PW, 512ch|14x14x512|
|conv_ds_12|DW + PW, 512ch|14x14x512|
|conv_ds_13|DW stride=2 + PW, 1024ch|7x7x1024|
|conv_ds_14|DW + PW, 1024ch|7x7x1024|
|avg_pool_15|7x7 avg pool|1x1x1024|
|SpatialSqueeze|squeeze spatial dims|1024|
|fc_16|fully connected|num_classes|

用于检测/分割时，常见会取：

- conv_ds_6：stride 8，较高分辨率
- conv_ds_12：stride 16
- conv_ds_14：stride 32，语义最强

其中 conv_ds_xx 结构如下：DepthwiseConv: [[nn积木/DSC 深度可分离卷积|DSC 深度可分离卷积]]
```
DepthwiseConv 3x3 
→ BatchNorm
→ ReLU
→ PointwiseConv 1x1
→ BatchNorm
→ ReLU
```


# V2
- - V2 改成“先扩张通道，再 depthwise，再压回去”
- **Linear Bottleneck**
    - 最后压缩通道那层通常**不加 ReLU**

```
1x1 Expand  (1 * 1 卷积)
→ BN + ReLU6
→ 3x3 Depthwise
→ BN + ReLU6
→ 1x1 Project   (1 * 1 卷积)
→ BN
→ (可选 Residual Add)
```

注意：

- Expand：把通道数先放大，比如乘 6
- Project：再把通道压回较小维度
- Project 后面通常**没有激活函数**，这就是 linear bottleneck

**为什么要这么改**

**1. Inverted Residual**

- ResNet 传统 bottleneck 是：宽 → 窄 → 宽
- MobileNetV2 是反过来：**窄 → 宽 → 窄**
- 因为 depthwise conv 本身不负责强通道混合，先扩张到高维再处理，效果更好

**2. Linear Bottleneck**

- 如果在最后那个低维 bottleneck 上再用 ReLU，会丢失太多信息
- 所以 V2 在 projection 后去掉非线性，尽量保留信息

**MobileNetV2 整体结构**  
输入通常是 224x224x3 时，标准分类网络大致如下：

|Stage|Operator|t|c|n|s|
|---|---|---|---|---|---|
|0|Conv 3x3|-|32|1|2|
|1|Bottleneck|1|16|1|1|
|2|Bottleneck|6|24|2|2|
|3|Bottleneck|6|32|3|2|
|4|Bottleneck|6|64|4|2|
|5|Bottleneck|6|96|3|1|
|6|Bottleneck|6|160|3|2|
|7|Bottleneck|6|320|1|1|
|8|Conv 1x1|-|1280|1|1|
|9|Global Avg Pool + FC|-|classes|1|-|

这张表里：

- t：expand ratio，扩张倍数
- c：输出通道数
- n：重复次数
- s：这个 stage 第一个 block 的 stride

# V3

V3 的典型 block 是：


```
1x1 expand
→ BN + activation
→ 3x3 / 5x5 depthwise
→ BN + activation
→ optional SE
→ 1x1 project
→ BN
→ residual add
```

其中 activation 可能是：

- ReLU
- h-swish [[nn积木/激活函数汇总|激活函数汇总]]

SE 不是每个 block 都有，是部分 block 有。[[nn积木/SE 通道注意力|SE 通道注意力]]

---

**V2 vs V3 block 对比**

|项目|MobileNetV2|MobileNetV3|
|---|---|---|
|主体结构|inverted residual|inverted residual|
|expand|有|有|
|depthwise|3x3 为主|3x3 / 5x5|
|project|有 linear bottleneck|有 linear bottleneck|
|shortcut|有条件使用|有条件使用|
|注意力|无|部分 block 加 SE|
|激活|ReLU6|ReLU / h-swish|
|结构设计|人工设计|NAS + 人工优化|

---

**MobileNetV3-Large 大致结构**  
输入 224x224x3 时：

```
Conv 3x3, 16, stride=2, h-swish

BNeck 3x3, exp=16, out=16,  SE=False, NL=ReLU,    stride=1
BNeck 3x3, exp=64, out=24,  SE=False, NL=ReLU,    stride=2
BNeck 3x3, exp=72, out=24,  SE=False, NL=ReLU,    stride=1

BNeck 5x5, exp=72,  out=40, SE=True,  NL=ReLU,    stride=2
BNeck 5x5, exp=120, out=40, SE=True,  NL=ReLU,    stride=1
BNeck 5x5, exp=120, out=40, SE=True,  NL=ReLU,    stride=1

BNeck 3x3, exp=240, out=80,  SE=False, NL=h-swish, stride=2
BNeck 3x3, exp=200, out=80,  SE=False, NL=h-swish, stride=1
BNeck 3x3, exp=184, out=80,  SE=False, NL=h-swish, stride=1
BNeck 3x3, exp=184, out=80,  SE=False, NL=h-swish, stride=1

BNeck 3x3, exp=480, out=112, SE=True,  NL=h-swish, stride=1
BNeck 3x3, exp=672, out=112, SE=True,  NL=h-swish, stride=1

BNeck 5x5, exp=672, out=160, SE=True,  NL=h-swish, stride=2
BNeck 5x5, exp=960, out=160, SE=True,  NL=h-swish, stride=1
BNeck 5x5, exp=960, out=160, SE=True,  NL=h-swish, stride=1

Conv 1x1, 960
Global Avg Pool
Conv 1x1, 1280
Classifier

```

# V4

V4 整体上可以抽象成：

```
Stem
→ Stage 1: UIB blocks
→ Stage 2: UIB blocks
→ Stage 3: UIB blocks
→ Stage 4: UIB / Mobile MQA mixed
→ Stage 5: deeper UIB / attention
→ Head
```

## 新的block UIB:

UIB 的思想是：把 block 设计做成一个更灵活的模板，里面可以容纳不同风格的子结构。

你可以把它理解成一个更大的框架：

```
(optional) local spatial mixing
→ (optional) expand 1x1
→ depthwise / spatial mixing
→ (optional) project 1x1 →
residual add
```

论文里强调，UIB 可以统一几类结构：

- **IB**：传统 inverted bottleneck (类似V3中先扩展再投影的block)
- **ConvNeXt 风格块**
- **FFN 风格通道混合**
- **ExtraDW**：
	- `先来一次 depthwise 做局部空间建模 再进入 expand / depthwise / project 主体`
	- 更早地引入空间感受野

V2/V3/V4 的 block 对比

**V2**

```
1x1 expand
→ 3x3 depthwise
→ 1x1 project
```

**V3**

```
1x1 expand
→ 3x3/5x5 depthwise
→ SE
→ 1x1 project
```

**V4 / UIB**

```
(optional) extra depthwise
→ 1x1 expand
→ depthwise / local mixing
→ 1x1 project
→ (optional attention / residual)
```

- block 设计自由度更高
- 让 NAS 去搜索哪种组合更适合目标硬件

---

**7. Mobile MQA 是什么**  [[TODO]]

---

## NAS 搜索

V3 已经用了 NAS，但 V4 更进一步，强调：

- block 模板更丰富
- 搜索空间更合理
- 搜索目标不只是精度/FLOPs，而是实际硬件延迟（避免FLOPs 很低，但真实手机上不一定快）

**它和“正常训练”通常分两步**

**第一步：搜结构**

- 先搭一个大搜索空间 / supernet
- 快速评估很多候选结构
- 看谁在“精度 + 延迟”上更优
```
先搭一个包含很多候选 block 的 SuperNet
→ 反复采样子网络
→ 用共享权重快速评估
→ 同时考虑精度和目标硬件延迟
→ 两阶段缩小搜索空间
→ 选出最优结构
```

不是把每个模型都认真训完。

**第二步：训最终模型**

- 搜出一个最优结构后
- 再单独拿这个结构从头训练
- 用正式训练策略、蒸馏、数据增强等把精度拉满


