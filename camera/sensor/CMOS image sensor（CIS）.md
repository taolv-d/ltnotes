# 1. CIS 制造工艺

CIS 一个pixel的组成：光电二极管/晶体管、bayer滤镜、微透镜

![](<attachments/CMOS image sensor（CIS）-image-8.png>)

## CMOS工艺

![](<attachments/CMOS image sensor（CIS）-image-7.png>)



![](<attachments/CMOS image sensor（CIS）-image-5.png>)

## bayer滤镜制造：

1. 薄膜淀积，生成一层透明的材料，用于后续bayer滤镜染色

2. 旋涂光刻胶

3. 光刻，遮住不想染色的区域

4. 刻蚀，清除未被光刻区域的覆盖

5. 染色

## 微透镜制造：

1. 旋涂光刻胶

2. 光刻，每个pixel 上制造一个小圆柱

3. 刻蚀，清除多余的光刻胶

4. 加热融化小圆柱形成微透镜

# 2. CIS 成像基本原理

![](<attachments/CMOS image sensor（CIS）-image-6.png>)

## 为什么是硅，而不是其他半导体材料，如锗、氮化镓等？

1. 硅禁带宽度 (eV) 1.12，正好对应可见光光子能量（1.65-3.26 eV），光谱响应范围300-1100 nm

![](<attachments/CMOS image sensor（CIS）-image.png>)

* 量子效率（激发电子数/入射光子数）高，硅对可见光的吸收很强

* 禁带较宽，暗电流低，高温性能好

* 与CMOS工艺完美兼容

## 光电转换

1. 光子激发出电子-空穴对

![](<attachments/CMOS image sensor（CIS）-image-1.png>)

* 电子捕获与存储

光子激发出电子的概率也称为量子效率，由光激发产生的电子叫做光生电子或光电子。光子激发出电子会被像点下方的电场捕获并囚禁起来备用，如下图所示。这个电场的专业名称叫做“势阱”，后面会有专门讨论。

![](<attachments/CMOS image sensor（CIS）-image-2.png>)

## bayer滤镜与微透镜

为了能够区分颜色，人们在硅感光区上面设计了一层滤光膜，每个像素上方的滤光膜可以透过红、绿、蓝三种波长中的一种，而过滤掉另外两种，如下图所示。光生电子本身是没有颜色概念的，此图中把电子的颜色只是为了说明该电子与所属像点的关系。

![](<attachments/CMOS image sensor（CIS）-MK9hb2CVPoVRLSxuXApcGf9WnLe.jpg>)

![](<attachments/CMOS image sensor（CIS）-Zu9SbVej9onHV1xTo3cccmgenAy.jpg>)

其中感光膜的布局叫做Bayer Mosaic Color Filter Arrary，通常简写为Bayer CFA或CFA。

早期的工艺微透镜之间是存在无效区域的，为了提高光能量的利用率，人们会努力扩大微透镜的有效面积，最终实现了无缝的透镜的阵列。

![](<attachments/CMOS image sensor（CIS）-KReAbXrwyock17xnFpUchpfTnBe.png>)

从RAW 数据计算RGB 数据的过程在数学上是一种不适定问题（ill-posed problem），理论上有无穷多种方法，因此与其说是一种科学，不如说是一种艺术。

下面介绍一种最简单的方法。这个方法考虑3x3范围内的9个像素，为简单起见只考虑两种情形，即中心像素为红色和绿色，其它情形同理。

![](<attachments/CMOS image sensor（CIS）-BatJbYsPMoSaZxxfcLFcXIfVnQd.png>)

<span style="color: rgb(143,149,158); background-color: inherit">中心像素为R</span>

上述过程常称为Bayer Demosaic，或者Debayer，经过此操作之后，每个像素就包含了3个完整的颜色分量，如下图所示。

![](<attachments/CMOS image sensor（CIS）-VuiEbondTo59yPxfiUkcAx8FnHh.png>)



上述各种Bayer格式的共同特点是接受一种颜色而拒绝两种颜色，因此理论上可以近似认为光能量损失了2/3，这是非常可惜的。为了提高光能量的利用率，人们提出了RYYB的pattern，这是基于CMY三基色的CFA pattern，Cyan是青色（Red的补色），Magenta是品红（Green的补色），Yellow是黄色（Blue的补色）。目前这种特殊的Bayer pattern已经在华为P30系列和荣耀20手机上实现了量产。据华为终端手机产品线总裁何刚透露，为了保证RYYB阵列在调色方面的准确性，华为付出了整整3年的时间。

![](<attachments/CMOS image sensor（CIS）-IN5fbNZT4ogM9zxAC0ecASwcnib.png>)

## 读出电路

![](<attachments/CMOS image sensor（CIS）-image-3.png>)

## Rolling shutter

### Rolling shutter 的工作模式

![](<attachments/CMOS image sensor（CIS）-image-4.png>)

下图显示了一个像素的曝光过程。

1. 一个曝光过程从RESET开始，RESET信号保持一段时间后像素清零，恢复高电压

2. 像素自由积分，时间取决于用户设置的曝光时间

3. 像素采样，准备读出

![](<attachments/CMOS image sensor（CIS）-MMjNbzQQFoU976xhyc1cnqWZn2c.png>)

Rolling shutter 在空间和时间上的关系如下图所示。

![](<attachments/CMOS image sensor（CIS）-QdpwbGiArodySqxcdzlcHd0Dnld.jpg>)

### 果冻效应

![](<attachments/CMOS image sensor（CIS）-image-9.png>)

[5d0da8ac-23d0-11eb-9352-5e2febf96652.mp4](<attachments/CMOS image sensor（CIS）-5d0da8ac-23d0-11eb-9352-5e2febf96652.mp4>)

一般来说，RS效应存在三种表现形式，前两种属于画面畸变，合称果冻效应。

* 整体倾斜（skew），如下图车辆的例子

![](<attachments/CMOS image sensor（CIS）-TmzzbouproknlbxcbxjczFtxn6h.jpg>)

<span style="color: rgb(143,149,158); background-color: inherit">传送带上的电路板图像运动skew</span>

* 图像摇摆（wobble），如下图所示

在无人机、车载等应用中，camera本身随载具平台一起运动，平台的高频机械振动会对成像造成较大扰动，图像产生摇摆。即使在安防场景中，如果camera附近存在振动源（如空调电机）也会产生同样的问题。

![](<attachments/CMOS image sensor（CIS）-image-10.png>)

* 部分闪光（partial flash），如下图所示
  普通摄影闪光灯的闪光时间 通常只有几个毫秒，显著短于一帧图像的成像时间，因此只有一部分画面能够被闪光照亮。

![](<attachments/CMOS image sensor（CIS）-SusCbv0OSoqDpwxeM2XcYO6onJh.png>)

### 如何避免果冻效应

1. 机械快门

![](<attachments/CMOS image sensor（CIS）-image-18.png>)

* Global shutter

![](<attachments/CMOS image sensor（CIS）-image-17.png>)

## 工频闪烁

为了避免工频闪烁，曝光时间应设置为光源能量周期的整数倍。在中国，光源能量周期为10ms(交流电周期的1/2)，在美国则为8.3ms，调整曝光时间时要特别注意这一点。

![](<attachments/CMOS image sensor（CIS）-image-16.png>)

## 光学格式

![](<attachments/CMOS image sensor（CIS）-image-15.png>)

注意用sensor 封装后的对角线长度一定大于sensor 光敏阵列的对角线长度。比如标称的1/2" 格式，指的是sensor 封装后对角线长12.7mm，而实际的光敏阵列则为6.4mm\*4.8mm，对角线长8mm。这个定义方式源于最初的电视技术采用的阴极射线摄像管，如下图所示。考虑到元件替换是最为普遍的需求，普通用户只需要关注这种摄像管的外径尺寸，而并不关心其内部成像面的具体尺寸。

![](<attachments/CMOS image sensor（CIS）-image-11.png>)



# 3. Pixel 内部

## 主动像素与被动像素

1. 被动像素：

**问题：噪声大，先读出再放大，读出电路的噪声会被一起放大**

| ![](<attachments/CMOS image sensor（CIS）-image-12.png>) | ![](<attachments/CMOS image sensor（CIS）-image-13.png>) |
| ------------------------------------------------- | ------------------------------------------------- |

* 主动像素（3T有源像素）

![](<attachments/CMOS image sensor（CIS）-image-14.png>)

* 复位。使能RST给PN结加载反向电压，复位完成后撤销RST。

* 曝光。与Passive Pixel 原理相同。

* 读出。曝光完成后，RS会被激活，PN结中的信号被SF放大后读出。

* 循环。读出信号后，重新复位，曝光，读出，不断输出图像信号

## PDD (钳位光电二极管)与CDS (相关双采样)

### 3T结构的问题

1. **PD固有缺陷，电荷不会完全转移，造成图像拖影。产生100个电子，实际只有80个读出，剩下20个叠加到下一帧**

| ![](<attachments/CMOS image sensor（CIS）-DXqab9vmroSXp8xbLXhcaENxnph.png>) | ![](<attachments/CMOS image sensor（CIS）-Z4Vubm7OZozWi9xhCdccmlTUnZf.png>) |
| -------------------------------------------------------------------- | -------------------------------------------------------------------- |

* **复位电路噪声（kTC噪声），每次PD复位后的电压有差异，产生相同的电子数，但读出电压不一致**



### PDD

![](<attachments/CMOS image sensor（CIS）-image-29.png>)

### CDS

![](<attachments/CMOS image sensor（CIS）-image-27.png>)



![](<attachments/CMOS image sensor（CIS）-image-26.png>)

![](<attachments/CMOS image sensor（CIS）-image-28.png>)

<span style="color: rgb(143,149,158); background-color: inherit">pixel noise variation (a) 无CDS (b) 有CDS</span>

## 前照式、背照式、堆栈

| FSI                                               | BSI                                               |
| ------------------------------------------------- | ------------------------------------------------- |
| ![](<attachments/CMOS image sensor（CIS）-image-19.png>) | ![](<attachments/CMOS image sensor（CIS）-image-20.png>) |

**BSI工艺**

![](<attachments/CMOS image sensor（CIS）-image-21.png>)

**堆栈式**

![](<attachments/CMOS image sensor（CIS）-Hf5CbcC9boQTtPx68abcgcgenac.png>)

该sensor使用上下两层硅片，通过一定的机制绑定成3D结构。下图是SONY发布的实物照片。

![](<attachments/CMOS image sensor（CIS）-XdNfbWitUo70lpxhyghcI4eXnBg.png>)



# 4. CIS 特性

一个理想的sensor 应该具备以下一些特性

* 输出与输入恒成正比（无sensor噪声，只有信号本身的噪声）

* 输入输出均可以无限大

* 高灵敏度，小的输入激励大的输出

* 高帧率

* 高分辨率

* 低功耗

* 工艺简单

* 低成本

而实际的sensor只能是在一段有限的区间内保持线性响应，对于幅度过小或者过大的输入信号会不能如实地表示。

![](<attachments/CMOS image sensor（CIS）-EEPwb8aAroJfXOxUt27csJJmnuh.png>)

<span style="color: rgb(143,149,158); background-color: inherit">实际sensor的响应特性（简化模型）</span>

## 性能参数

![](<attachments/CMOS image sensor（CIS）-image-22.png>)

<table><colgroup><col width="100"><col width="371"><col width="339"></colgroup>
<thead>
<tr>
<th>量子效率</th>
<th>特定波长下单位时间内产生的平均光电子数与入射光子数之比</th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td>满阱容量</td>
<td>一个像素的势阱最多能够容纳多少个光生电子，消费类的sensor一般以2000~4000较为常见，此值越大则sensor的动态性能越好。</td>
<td></td>
</tr>
<tr>
<td>噪声</td>
<td>在处理过程中设备自行产生的<a href="https://link.zhihu.com/?target=https%3A//baike.baidu.com/item/%25E4%25BF%25A1%25E5%258F%25B7"><span style="color: rgb(36,91,219); background-color: inherit">信号</span></a>，这些信号与输入信号无关</td>
<td></td>
</tr>
<tr>
<td>Black level</td>
<td>Sensor 无光时 输出的信号强度</td>
<td></td>
</tr>
<tr>
<td>信噪比</td>
<td>信噪比是一个电子设备或者电子系统中信号与噪声的比例。信噪比很低，单纯的放大操作并不增加任何有用的信息，无助于改善图像质量<br /></td>
<td><img src="attachments/CMOS%20image%20sensor%EF%BC%88CIS%EF%BC%89-image-23.png" alt=""><img src="attachments/CMOS%20image%20sensor%EF%BC%88CIS%EF%BC%89-image-24.png" alt=""></td>
</tr>
<tr>
<td>动态范围 DR<br /></td>
<td>动态范围表示图像中所包含的从“最暗”至“最亮”的取值范围。根据ISO15739的定义，“最亮”指的是能够使输出编码值达到特定“饱和值”的亮度；而“最暗”指的是图像信噪比下降至1.0时的亮度。<br />人眼：80~100dB<br />消费级sensor: 50~70dB<br />车规级sensor: 120dB（甚至不需要调节曝光）<br /></td>
<td><img src="attachments/CMOS%20image%20sensor%EF%BC%88CIS%EF%BC%89-image-25.png" alt=""><img src="attachments/CMOS%20image%20sensor%EF%BC%88CIS%EF%BC%89-image-30.png" alt=""></td>
</tr>
<tr>
<td>串扰</td>
<td>光子是可以在硅片中穿透一定的距离的，影响其他像素的信号<br /></td>
<td><img src="attachments/CMOS%20image%20sensor%EF%BC%88CIS%EF%BC%89-G0o7bMCIIocerfxhAU0cl7Blnpe.png" alt=""></td>
</tr>
<tr>
<td>灵敏度</td>
<td>CMOS sensor 对入射光功率的响应能力用灵敏度参数衡量，常用的定义是在1μm^2单位像素面积上，标准曝光条件下(1Lux照度，F5.6光圈)，在1s时间内积累的光子数能激励出多少mV的输出电压。<ul>
<li>在图像噪声水平接近的情况下，灵敏度高的sensor图像亮度更高、细节更丰富</li>
<li>在图像整体亮度接近的情况下，灵敏度高的sensor噪声水平更低，图像画质更细腻</li>
</ul></td>
<td><img src="attachments/CMOS%20image%20sensor%EF%BC%88CIS%EF%BC%89-AZZybRFdcovUqHxya8JcLyjInxd.jpg" alt=""></td>
</tr>
<tr>
<td>填充系数</td>
<td>接收光的面积/总面积<br />微透镜可以将填充系数提高到90%<br /></td>
<td><img src="attachments/CMOS%20image%20sensor%EF%BC%88CIS%EF%BC%89-image-31.png" alt=""><img src="attachments/CMOS%20image%20sensor%EF%BC%88CIS%EF%BC%89-image-32.png" alt=""></td>
</tr>
<tr>
<td>像素尺寸</td>
<td>大的像素通常可以容纳更多的电子，因此可以表示更大的信号变化范围，这个指标称为sensor的动态范围。在极低照度下，大的像素更容易捕获到少量的光子，也就是低照度性能会更好。</td>
<td><img src="attachments/CMOS%20image%20sensor%EF%BC%88CIS%EF%BC%89-image-33.png" alt=""></td>
</tr>
<tr>
<td>主光线角CRA<br /></td>
<td>主光线角（Chief Ray Angle, CRA)是衡量sensor 收集入射光能量的一个主要参考指标<br /></td>
<td><img src="attachments/CMOS%20image%20sensor%EF%BC%88CIS%EF%BC%89-image-34.png" alt=""></td>
</tr>
</tbody>
</table>

**CRA:**

镜头CRA

| 长焦                                                | 广角                                                | 超广角                                               |
| ------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| ![](<attachments/CMOS image sensor（CIS）-image-35.png>) | ![](<attachments/CMOS image sensor（CIS）-image-36.png>) | ![](<attachments/CMOS image sensor（CIS）-image-37.png>) |

Sensor CRA

![](<attachments/CMOS image sensor（CIS）-image-38.png>)

## CIS的噪声

![](<attachments/CMOS image sensor（CIS）-image-39.png>)

### 噪声的数学模型

下图以传递函数的形式总结了CMOS sensor 光、电转换模型以及几种主要噪声的数学模型。

| ![](<attachments/CMOS image sensor（CIS）-JbfTb7K0OoBlz5xuGW1cuWRLncb.png>) | ![](<attachments/CMOS image sensor（CIS）-IUlab3VYUoMU9dxfHtCcBxqmnIF.png>) |
| -------------------------------------------------------------------- | -------------------------------------------------------------------- |

### 泊松分布

**泊松分布的标准差是信号的平方根**

![](<attachments/CMOS image sensor（CIS）-WEN4b5jASo9iYlx9TfQc8wktnGh.png>)

### 噪声的分类

![](<attachments/CMOS image sensor（CIS）-image-43.png>)

1. 固定模式噪声(FPN)

| DSNU       | 完全无光条件下，因像素电路制造偏差导致各像素输出的数字值存在固定差异，表现为图像上的固定斑点或网格                                   |                                                                      |
| ---------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| PRNU       | 光响应非均匀性<br />                                                                       | ![](<attachments/CMOS image sensor（CIS）-CNXcbGyoPoo2qexxZE3cVq9TnWf.png>) |
| Column FPN | 列级电路不均匀（列级放大器、列级噪声抑制电路等、列级ADC）。列扫描电路引入的噪声（解码器、列选开关等）<br />                          | ![](<attachments/CMOS image sensor（CIS）-YypmboDFJo1WU6xFFCjcpI5NnWd.jpg>) |
| Row FPN    | 供电波动等（列级放大器/噪声抑制电路的供电随时间波动）                                                         | ![](<attachments/CMOS image sensor（CIS）-LYr5bowHwo14T9xaD9QcFHx7nyc.png>) |
| Pixel FPN  | 坏像素、瑕疵像素<br />Hot pixel（暗电流过大）<br />Dark pixel（像素击穿）<br />Weak pixel (像素性能差异)<br /> | ![](<attachments/CMOS image sensor（CIS）-image-40.png>)                    |

* 时域噪声

| 暗散粒噪声<br />                     | 热电子随机运动形成的信号，没有光照射时也会有信号。**温度越高，噪声越强**,一般的规律是温度每升高8°C暗电流翻一倍<br />                                                                                                                     | ![](<attachments/CMOS image sensor（CIS）-image-41.png>)                                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 光子散粒噪声                          | 在微观尺度下，光子流到达传感器的行为在时间和空间上都是不均匀的，整体上其统计规律符合泊松分布。光子散粒噪声与被测信号强度有关。                                                                                                                       | ![](<attachments/CMOS image sensor（CIS）-image-45.png>)                                                                                            |
| 读出噪声                            | 放大器引入噪声<br />量化噪声：<br />模拟信号转换为数字信号时，ADC 精度不足造成的精度损失引入的噪声。通常ADC精度设计比系统总噪声高一点点                                                                                                         |                                                                                                                                              |
| 复位噪声<br />（PDD+CDS）技术已经解决<br /> | 卷帘曝光方式需要在先对势阱复位，将势阱中自由积累的电荷全部释放，为后续的读出准备。但是由于暗电流的存在，每次复位后都会残留一些大小随机的噪声信号，即复位噪声，其大小与像素结构、芯片温度、PN结电容有关，因此也称为kTC噪声。<br />像素的复位是需要一定时间的。定量的研究表明，即使是采用较大的复位电流，一般也需要1ms以上的时间才能将电荷释放干净<br /> | ![](<attachments/CMOS image sensor（CIS）-DXqab9vmroSXp8xbLXhcaENxnph-1.png>)![](<attachments/CMOS image sensor（CIS）-Z4Vubm7OZozWi9xhCdccmlTUnZf-1.png>) |
| 1/f 噪声                          | 也称flicker noise（闪烁噪声） 或pink noise（粉红噪声），它广泛存在于半导体器件中。在低频的时候1/f噪声一般显著高于电散粒噪声                                                                                                           |                                                                                                                                              |

## HDR

| 长短帧融合<br /> | 运动伪影<br />                                                | ![](<attachments/CMOS image sensor（CIS）-image-44.png>)                                                                                        |
| ----------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 行交替曝光       | 损失分辨率<br />                                               | ![](<attachments/CMOS image sensor（CIS）-image-46.png>)                                                                                        |
|  Dual-diode |                                                           | ![](<attachments/CMOS image sensor（CIS）-A7GebQoJ5oUNX4xa9YjcVRJ3nMh.png>)                                                                     |
| 对数响应        | Lin-Log 函数响应，依赖特别的像素设计，LDR区间为线性响应，HDR区间变为log规律响应，缺点是FPN较大 | ![](<attachments/CMOS image sensor（CIS）-image-42.png>)                                                                                        |
| DCG<br />   | 使用两个不同增益的读出电路，读出两个不同强度的信号<br />高增益保留暗区<br />低增益保良亮区       | ![](<attachments/CMOS image sensor（CIS）-HzjUbdScJopxMExJjTJcr6Adntg.png>)![](<attachments/CMOS image sensor（CIS）-AqHKbI0aZovKk8xpzBScAqDCnZd.png>) |
| LOFIC       | 每个像素都配置一个较大的电容用于收集因饱和而溢出的电荷                               | ![](<attachments/CMOS image sensor（CIS）-HILwbUOeGooIqlxyF5GcDEwonTf.png>)                                                                     |

# 5. 图像伪影

## 摩尔纹与迷宫格



![](<attachments/CMOS image sensor（CIS）-image-47.png>)

| 摩尔纹 | 当sensor像素阵列的空间频率低于信号本身的频率时就会发生频谱混叠<br />                                                                                        | ![](<attachments/CMOS image sensor（CIS）-image-48.png>)![](<attachments/CMOS image sensor（CIS）-F8hGbTA3Bo1ZqbxzdMjcrFDQnec.jpg>) |
| --- | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 迷宫格 | 当图像中存在平行线时，经常会出现平行线的边缘刚好覆盖半个像素的情况，后续的ISP算法必须要决定这个像素到底是属于A物体还是B物体。就像抛硬币一样，ISP会有一半的概率做出相反的决定。如果平行线的间距足够小，则会有相当的概率发生两条平行线搭接到一起的情况。 | ![](<attachments/CMOS image sensor（CIS）-F4pgbLxn7oxbf0xVZFncHl4vnZg.jpg>)                                                  |

## 紫边 (Purple Fringing)



![](<attachments/CMOS image sensor（CIS）-T7A7bGC7FoiVMkxLgOmctP2Knwe.jpg>)

![](<attachments/CMOS image sensor（CIS）-PayBbl0GGogQrHxqXFicoM3Znrd.jpg>)

![](<attachments/CMOS image sensor（CIS）-Bj8wbp3aXotgZmx5jpncccthnpa.jpg>)

去紫边镜头原理

![](<attachments/CMOS image sensor（CIS）-UZombFC2loHVYTxDAQ3c997vndb.jpg>)

## 眩光与鬼影 (Flare)

由强光造成的照片发白、形成光晕的现象称作眩光(Flare)。当光源在画面上移动时，光晕也会在画面上漂浮游动，忽上忽下，忽左忽右，有如幽灵一般，所以称之为鬼影。我们知道镜头是由多枚镜片构成的，而镜片则是采用玻璃或塑料等材料制造，如果不进行特殊处理，镜片表面会反射大约5%左右的入射光线。当强光进入镜头时，各枚镜片的表面反射的光线会在镜头和摄像机内部多次反射，最终在sensor上映射出一连串的光晕，这就是眩光和鬼影产生的原因。

第一，**镜头光学表面的反射**。按照菲涅尔定律，凡是存在折射系数突变的地方就会存在反射，也就是只要镜头材料的折射系数不等于1（也就是空气的折射系数）就会存在一定程度的反射，反射的程度与入射光线的角度有关。在镜片上镀增透膜可以降低反射系数，一般来说多层镀膜的效果要优于单层镀膜，但是镀膜会提高镜片的成本，同时也存在收益边界，即使是最优秀的镀膜技术也无法完全抑制菲涅尔反射。

第二，**镜片侧面的反射**。在镜头设计过程中，设计师往往只是考虑镜片前后两个光学表面对光线的折射，很少考虑镜片的侧面（也就是圆柱面）对光的影响。有些设计精良的镜头会把镜片侧面涂成全黑，阻断杂散光在镜头内传播的路径。因为世界上没有反射系数等于0的全黑材料，所以也只能部分阻断。

第三，**光学表面的瑕疵**。如划痕，微粒，边角处的崩碎，都会反射/折射额外的光线。

第四，**镜筒内部的散射**。尽管大多数镜头内部都采用黑色消光材料制作，有些还在内表面上加工了消光螺纹，但是仍旧无法完全消除杂散光的影响。

![](<attachments/CMOS image sensor（CIS）-LkcXbxCgaouogTxbXpacRCeonpw.jpg>)

![](<attachments/CMOS image sensor（CIS）-C0ynb2bHtonIs6xJwTmcoIvlnef.png>)

## &#x20;等高线效应 (Contouring)

8 位二进制数最多可以表示256个不同的灰度等级。当sensor的精度低于8位时，sensor的输出不能很好地表示渐变的颜色，就会出现下图所示的等高线效应。这种效应在拍摄白墙等单色背景时尤其明显。

另一方面，虽然现在的主流sensor都是10位或者12位的，但是最终图像的存储格式仍然是以8位为基础的，比如H.264/H.265图像编码器需要YUV420作为输入格式，其中亮度分量Y用8位数据表示，这就会不可避免地导致contour效应。

为了缓解contour效应，一种常用的办法是在量化过程中人为地施加随机噪声，噪声的作用是模糊原本清晰的量化边界，使灰度渐变显得更加平滑，这种技术叫做抖动（dithering），可以有效地缓和contour。另一种办法则更加简单直接，即用10\~12位数表示YUV，从源头上避免精度损失。这种方法涉及到编解码、存储、传输、显示整个链条的设备升级，因此必然是一个缓慢的过程。

![](<attachments/CMOS image sensor（CIS）-WU9db3lauosejgxtmdvcdD0inPf.png>)

