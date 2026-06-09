[[../../TODO|TODO]] 这篇2024年的文章需要进一步整理
## 基本概念

### 名词解释
![[attachments/2024-01-07-13-38-47-image.png]]
**optical axis** 光轴

**optical center** 光心

**aperture** 光圈

**focal length（*f*）** 焦距

**** 

### 光圈与F数
![[attachments/2024-01-07-13-43-49-image.png]]
**光圈与NA**
![[attachments/2024-01-07-13-44-26-image.png]]
****

**aperture stop 与 field stop**
![[attachments/2024-01-07-13-47-36-image.png]]
**AS** 决定了物方发出光线的直径

**FS** 决定了系统视野大小

****

### 入瞳 出瞳

![[attachments/2024-01-07-13-49-39-image.png]]
**入瞳**从物方向镜头里面看到的光圈的虚像
**出瞳**从像方向镜头里面看到的光圈的实像
更深入的介绍见：[[Entrance Pupil and Exit Pupil]]

****

### 景深

[景深与焦深 | 爱特蒙特光学](https://www.edmundoptics.cn/knowledge-center/application-notes/imaging/depth-of-field-and-depth-of-focus/)

![[attachments/Pasted image 20260609190802.png]]

更改镜头的f/#会更改景深，如图3所示。对于图3中所示的每种配置，都有两束光线。黑色虚线代表的光束显示了其从物体移到镜头系统时信息的分布情況。随着物体不断远离最佳焦点位置（虚线相交处），物体细节会移动到更广的锥形区域。锥形分布得越宽，该距离处来自镜头的信息与其周围的所有其他信息的界限越模糊。镜头的f/#可控制锥形扩展的速度，进而控制在给定距离实际上有多少信息或细节模糊成一片。
图中还有一個红色锥形，用角度表示了系统的分辨率。其中，两个锥形的线条相交处可界定整个景深范围。f/#越低，黑色虚线扩展得越快，景深越低。

随着细节变小，图3a和3b中的光束一起靠近，加快了这种效果。最后，f/#增加太多会由于达到镜头的衍射极限而造成较小的细节变得模糊，因为镜头的极限分辨率与f/#成反比。此限制意味着，虽然增加f/#总会增加景深，但可解析（即使在最佳焦点下）的特征尺寸也会变大。有关衍射极限及其与f/#的关系的更多信息，请参见[衍射限制](https://www.edmundoptics.cn/knowledge-center/application-notes/imaging/diffraction-limit/)。在此区域内利用短波长的确情有可原，并可通过多种方式挽回分辨率损失。有关更改波长影响系统性能的更多信息，详见[波长对性能的影响](https://www.edmundoptics.cn/knowledge-center/application-notes/imaging/wavelength-effects-on-performance/)。

一般来说，当镜头在较短工作距离聚焦时，大锥角会导致锥形在最佳焦点两侧很快发散，造成景深有限。对于在较长工作距离下聚焦的物体，光束跃迁率会下降，并且景深会增加。


**图3**是一张原理示意图，标题为“**高和低f/#镜头的景深的几何表示法**”。它用几何光学的锥形光束模型，直观解释了**光圈大小（f/#）如何影响景深**。

这张图通常由两个部分（a和b）对比组成，核心逻辑如下：

### 图解核心：光束发散速度决定景深

图中用两束光线代表两种物理过程：

1. **黑色虚线光束**：代表从物体上一个点发出的光线，通过镜头后汇聚再发散。它的**发散速度**取决于f/#（光圈）。
    
2. **红色锥形光束**：代表系统的**分辨率极限**（能分辨的最小细节对应的角度）。
    

### 关键对比：低f/# vs. 高f/#

|配置|图例特征|物理含义|对景深的影响|
|---|---|---|---|
|**低f/# (大光圈)**  <br>（例如 f/2.8）|黑色虚线光束的锥角**很宽**，离开焦点后**迅速扩展**。|光线发散极快，物体稍微偏离焦点，光斑就会快速变大并模糊。|**景深很浅**  <br>（只有焦点附近一小段范围清晰）|
|**高f/# (小光圈)**  <br>（例如 f/8）|黑色虚线光束的锥角**很窄**，离开焦点后**缓慢扩展**。|光线发散缓慢，物体在更大范围内移动时，光斑尺寸变化不大。|**景深很深**  <br>（清晰范围明显扩大）|

### 图中如何界定“景深范围”？

景深范围在图中由**黑色虚线光束**和**红色分辨率光束**的**交点**来界定：

- 在交点以内，系统分辨率足够分清细节；
    
- 超出交点，模糊圈超过允许范围，就超出了可用景深。
    

### 理解图3的关键结论

结合图中信息，可以得出两个重要权衡：

1. **光圈与景深的反向关系**：f/# 越小（光圈越大），景深越浅；f/# 越大（光圈越小），景深越深。
    
2. **景深与分辨率的矛盾**：虽然提高 f/# 能增加景深，但图中也暗示了极限——f/# 太高时，**衍射极限**会使整体分辨率下降（即使焦点处的细节也会变模糊）。所以，不能为了景深无限制缩小光圈。
    

### 实际应用联想

- **需要“虚化背景”拍人像时**：就是选择**低f/#**（如f/1.4），对应**图3a**，获得浅景深。
    
- **需要“前后都清晰”拍风景或微距时**：会选择**高f/#**（如f/11），对应**图3b**，获得深景深。


![[attachments/2024-01-07-13-51-39-image.png]]
****

### 视野

![[attachments/2024-01-07-13-52-36-image.png]]
****

### 正透镜与负透镜

![[attachments/2024-01-07-13-53-14-image.png]]
****

**shape factor $\sigma$**

$$
\sigma = \dfrac{R_2+R_1}{R_2-R_1}
$$
![[attachments/2024-01-07-13-58-52-image.png]]


离轴处会早成焦距变化，引发球差
![[attachments/2024-01-07-13-58-31-image.png]]
****
### CRA

CRA 的更多描述见 [[CRA]]
CRA mismatch 的分析见 [[CRA mismatch]]

成像面上主光线与光轴的夹角，包括 lens CRA 和 sensor CRA
![[attachments/2024-01-07-14-04-59-image.png]]

---

## 镜头材料

天然材料  玻璃  塑胶

折射、反射、吸收

材料选择可以避免**轴向色差**

### 阿贝数

在光学和透镜设计中，阿贝数又称透明材料的 V 数或常数，是材料色散（折射率随波长的变化）的近似测量值，**V 值高表示色散低**

### 制造工艺

a 粗胚 b 抽氧 c 充氮气 d 加热软化玻璃 e 加压 f 脱模
![[attachments/2024-01-07-14-12-54-image.png]]
非球面玻璃制造
![[attachments/2024-01-07-14-15-37-image.png]]
塑胶镜头制造
![[attachments/2024-01-07-14-16-44-image.png]]
### 轴向色差

原因
![[attachments/2024-01-07-14-18-30-image.png]]
使用高低阿贝数玻璃消色差
![[attachments/2024-01-07-14-19-10-image.png]]
满足该公式![[attachments/2024-01-07-14-19-41-image.png]]
### BR lens

BR镜片是采用了BR光学元件（蓝色光谱折射光学元件）的复合镜片。BR光学元件具有能大幅折射蓝色光（短波长光）的特性，可实现更理想的色像差补偿效果。

ref [佳能（中国）－ RF/EF镜头 － 技术介绍 － BR镜片 (canon.com.cn)](https://www.canon.com.cn/product/ef/info/info12.html)
![[attachments/2024-01-07-14-22-36-image.png]]

![[attachments/2024-01-07-14-22-45-image.png]]
## 矩阵光学

将光学元件用矩阵表达

## MTF

点扩散函数 PSF 见 [[PSF]]

### MTF

#### contrast MTF

由低频到高频评价振幅的大小
![[attachments/2024-01-07-14-44-41-image.png]]
#### log F

X 方向频率越来越高

Y 方向对比度有变化

![[attachments/2024-01-07-14-47-19-image.png]]
#### 棋盘格

$MTF=\dfrac{mean(br\%)-mean(dk\%)}{mean(br\%)+mean(dk\%)}$

![[attachments/2024-01-07-14-48-32-image.png]]
不能区分S/T方向

#### SFR
这里有更详细的介绍：[[../evaluation/MTF|MTF]]

#### 其他chart

SFR chart、SFRplus

siemens star(方波 正弦波)

Texture MTF 原点图

USAF1951

ISO12233

### MTF 单位

![[attachments/2024-01-07-15-02-30-image.png]]
![](attachments/2024-01-07-15-02-13-image.png)

## 光学像差

### 波前

理想光学系统的波前是理想的球面波，实际光学系统是不规则的波前

![[attachments/2024-01-07-15-08-43-image.png]]
### 像差

![[attachments/2024-01-07-15-13-44-image.png]]
球差：在轴和离轴区域聚焦能力不同（使用非球面透镜）

![](https://pic2.zhimg.com/80/v2-43dc3bf4300829fce18c67cf3cff2dd9_720w.webp)

![](https://pic1.zhimg.com/80/v2-a2e95690e7c20fb26f32f3a26c12ed4c_720w.webp)

![](https://pic1.zhimg.com/80/v2-cc4a1ec90afb7da816be065a68294cd0_720w.webp)

慧差：聚焦能力不同，但在Y方向分布（减小光圈改善）

![](https://pic3.zhimg.com/v2-dc1b1b04794c613759756cf07d611302_r.jpg)

![](https://pic4.zhimg.com/v2-e087f715145dfce2be93f8e0bf751c47_r.jpg)

色差：不同波长聚焦能力不同

像散：透镜X Y 方向聚焦能力不同

![](https://pic1.zhimg.com/v2-34cb26d8ed98a70ac77706ae3fb8bf60_r.jpg)

![](https://pic4.zhimg.com/80/v2-01e3c4637082ed53125f4cd114e7fbef_720w.webp)

场曲：成像面的聚焦点不在平面，而是球面或曲面

![](https://pic2.zhimg.com/80/v2-e93d1f4ff5043ca725b7ba6a96814ce1_720w.webp)

畸变：镜头不同半径处放大率不一致

![](https://pic2.zhimg.com/v2-852c9f43b6b5b3bec9f12763671e4625_r.jpg)

畸变与镜片类型以及主点相对镜片的前后位置有关。

![](https://pic1.zhimg.com/v2-0d20105fbb89016b1a8659a367956550_r.jpg)

虽然相对罕见，也有两者同时存在的复合形畸变，俗称八字胡（mustache）畸变，常出现在超广角镜头上。

镜头组合构成上，镜片对称的分置光圈两侧，畸变比较少；非对称构成的镜片，则经常发生。另外，变焦镜头的畸变在广角区为桶形，望远区为枕形（因变焦的不同，歪曲像差的特性稍微不同）。采用非球面镜片的变焦镜头，由于非球面镜片有消除歪曲像差的功能，矫正效果相当良好。畸变是通过镜头中心的主光线异常折射所引起的，**因此不论如何缩小光圈，都不能获得改善**。

### 赛德尔相差理论

用高阶多项式拟合波前？

![[attachments/2024-01-07-15-27-22-image.png]]
## 畸变 distortion

1、畸变是唯一一种不会造成解析力下降的相差

2、畸变与波长无光

可以在测试卡上加反向畸变来评价

外视场角放大率(y2/h2) > 内视场角放大率(y1/h1)  枕形畸变

外视场角放大率(y2/h2) < 内视场角放大率(y1/h1) 枕形畸变

![[attachments/2024-01-07-15-33-36-image.png]]
### 畸变测量

SMIA TV
![[attachments/2024-01-07-15-35-28-image.png]]
Traditional TV
![[attachments/2024-01-07-15-35-49-image.png]]
Lens geometric distortion
![[attachments/2024-01-07-15-36-36-image.png]]
DXO
![[attachments/2024-01-07-15-37-13-image.png]]
### 畸变矫正

二阶矫正模型
![[attachments/2024-01-07-15-39-20-image.png]]
高阶矫正模型
![[attachments/2024-01-07-15-39-53-image.png]]
## 鱼眼
![[attachments/2024-01-07-15-43-55-image.png]]
### 球面投影模型
![[attachments/2024-01-07-16-27-47-image.png]]

![[attachments/2024-01-07-16-31-38-image.png]]
![[attachments/2024-01-07-16-32-06-image.png]]

### 五种种投影模型

要拍摄的原始隧道，镜头从隧道内部中心向左墙拍摄。
![[attachments/2024-01-07-15-52-32-image.png]]

| A                                                                             | B                                              | C                                              | D                                                                                                          | E                                                          |
| ----------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Rectilinear                                                                   | Stereographic                                  | Equidistant                                    | Equisolid angle                                                                                            | Orthographic                                               |
| ![[attachments/2024-01-07-15-51-27-image.png]]                                | ![[attachments/2024-01-07-15-56-13-image.png]] | ![[attachments/2024-01-07-15-56-16-image.png]] | ![[attachments/2024-01-07-15-56-20-image.png]]                                                             | ![[attachments/2024-01-07-15-56-23-image.png]]             |
| $r=f\tan \theta$                                                              | $r=2f\tan \dfrac{\theta}{2}$                   | $r=f\theta$                                    | $r=2f\sin\dfrac{\theta}{2}$                                                                                | $r=f\sin \theta$                                           |
| 工作原理与针孔摄像机类似。直线保持笔直（无失真）。$\theta$ 必须小于 90°。光圈角与光轴对称，必须小于 180°。大孔径角设计难度大，价格也高。 | 保持角度。这种制图方式是摄影师的理想选择，因为它不会过多压缩边缘物体。            | 保持角距离。适用于角度测量（如星图）。                            | 保持表面关系。每个像素所占的实角相等，或单位球面上的面积相等。看起来像一个球上的镜像，最佳特效（不复杂的距离），适合面积比较（云层等级测定）。这种类型很受欢迎，但它会压缩边缘物体。这类镜头的价格较高，但并不极端。 | 保持平面照度。看起来像一个球体，周围环境位于 < 最大。180° 光圈角。图像边缘附近高度失真，但中心图像压缩较小。 |


### 四种球面投影模型的效果
![[attachments/2024-01-07-16-34-24-image.png]]
![[attachments/2024-01-07-16-35-54-image.png]]



![[attachments/2024-01-07-16-35-43-image.png]]

![[attachments/2024-01-07-16-36-43-image.png]]


![[attachments/2024-01-07-16-36-18-image.png]]
### 鱼眼镜头的应用

监控：
![[attachments/2024-01-07-16-38-39-image.png]]
全景拼接：
![[attachments/2024-01-07-16-40-10-image.png]]

![[attachments/2024-01-07-16-40-03-image.png]]

## local blur

### 成因

原因：镜头倾斜或偏移（镜片安装、音圈电机对焦时不同区域力不同、螺纹调焦没拧好）
![[attachments/2024-01-07-16-05-35-image.png]]

![[attachments/2024-01-07-16-05-03-image.png]]

![[attachments/2024-01-07-16-05-21-image.png]]
### 表现

MTF曲线非中心对称
![[attachments/2024-01-07-16-06-09-image.png]]
## 光学镀膜

### 镀膜工艺比较

sol-gel 浸泡
![[attachments/2024-01-07-16-41-41-image.png]]

PVD

下方加热盘将蒸镀材料加热挥发后凝结到上方的材料
![[attachments/2024-01-07-16-42-56-image.png]]

### 光学镀膜机台
![[attachments/2024-01-07-16-45-09-image.png]]

### AR coating (抗反射)
![[attachments/2024-01-07-16-45-59-image.png]]

原理：破坏性干涉（两个放射光光程差为$\lambda/2$,镀膜厚度为$\lambda/4$）
![[attachments/2024-01-07-16-47-10-image.png]]

多层镀膜：
![[attachments/2024-01-07-16-49-21-image.png]]

### Optical Density (OD滤镜) ND filter

$OD = \log10(I_0/I)$

$I_0$入射光，$I$出射光
![[attachments/2024-01-07-16-52-00-image.png]]

### IR cut filter
![[attachments/2024-01-07-16-52-24-image.png]]

白天需要IR,晚上可以不要
![[attachments/2024-01-07-16-52-59-image.png]]

### UV cut filter

用于改善紫边
![[attachments/2024-01-07-16-53-26-image.png]]

## 偏振

### 线偏振、圆偏振、椭圆偏振

#### 线偏振

无偏振光，经过偏振片，衰减为一半

线偏振光衰减根据夹角计算
![[attachments/2024-01-07-16-55-20-image.png]]

#### 圆偏振

圆偏振可以认为是一种特殊的椭圆偏振
![[attachments/2024-01-07-16-56-56-image.png]]

### 偏振应用

#### 消反射光
![[attachments/2024-01-07-17-00-12-image.png]]

#### 偏振分光棱镜 双折射晶体

用两种不同偏振态的晶体组合
![[attachments/2024-01-07-16-59-02-image.png]]

![[attachments/2024-01-07-16-59-32-image.png]]


#### 相位延迟 半波片 四分之一波片 光学低通滤波器
![[attachments/2024-01-07-17-02-15-image.png]]

应用：光学低通滤波器（消摩尔纹）
![[attachments/2024-01-07-17-03-58-image.png]]

#### LCD 显示器

利用液晶改变偏振态，调节光线强弱
![[attachments/2024-01-07-17-04-54-image.png]]

#### 3D glass
![[attachments/2024-01-07-17-05-49-image.png]]
