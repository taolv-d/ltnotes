---
type: note
status: review
tags:
  - camera
  - optics
rating: 0
create: 2026-05-12
update: 2026-08-30
---
# 光学系统
下图是最简答的单透镜系统。其中
**optical axis** 光轴
**optical center** 光心
**aperture** 光圈
**focal length（*f*）** 焦距
![[attachments/2024-01-07-13-38-47-image.png|401]]
## 光圈
**光圈与F数**
光圈越大，虚化效果越好
![[attachments/2024-01-07-13-43-49-image.png|431]]
**光圈与NA**
![[attachments/2024-01-07-13-44-26-image.png]]
**孔径光阑与视场光阑**
- **AS** 决定了物方发出光线的直径
- **FS** 决定了系统视野大小
![[attachments/2024-01-07-13-47-36-image.png]]
**入瞳与出瞳**
**入瞳**从物方向镜头里面看到的光圈的虚像
**出瞳**从像方向镜头里面看到的光圈的实像
更深入的介绍见：[[Pupil/Entrance Pupil and Exit Pupil]]
![[attachments/2024-01-07-13-49-39-image.png]]
## 景深
### 景深与F/#的关系
此次内容摘自[景深与焦深 | 爱特蒙特光学](https://www.edmundoptics.cn/knowledge-center/application-notes/imaging/depth-of-field-and-depth-of-focus/)
![[attachments/Pasted image 20260609190802.png|611]]

上图中：
1. **黑色虚线光束**：代表从物体上一个点发出的光线，通过镜头后汇聚再发散。它的**发散速度**取决于f/#（光圈）。
2. **红色锥形光束**：代表系统的**分辨率极限**（能分辨的最小细节对应的角度）。
3. **黑色虚线光束**和**红色分辨率光束**的**交点**来界定：
	- 在交点以内，系统分辨率足够分清细节；
	- 超出交点，模糊圈超过允许范围，就超出了可用景深。
4. **景深与分辨率的矛盾**：f/# 太高时，**衍射极限**会使整体分辨率下降（即使焦点处的细节也会变模糊）。

| 配置             | 图例特征                           | 物理含义                         | 对景深的影响                        |
| -------------- | ------------------------------ | ---------------------------- | ----------------------------- |
| **低f/# (大光圈)** | 黑色虚线光束的锥角**很宽**，离开焦点后**迅速扩展**。 | 光线发散极快，物体稍微偏离焦点，光斑就会快速变大并模糊。 | **景深很浅**  <br>（只有焦点附近一小段范围清晰） |
| **高f/# (小光圈)** | 黑色虚线光束的锥角**很窄**，离开焦点后**缓慢扩展**。 | 光线发散缓慢，物体在更大范围内移动时，光斑尺寸变化不大。 | **景深很深**  <br>（清晰范围明显扩大）      |
### 景深的计算
![[attachments/2024-01-07-13-51-39-image.png|664]]
## 视野

![[attachments/2024-01-07-13-52-36-image.png|700]]
# 透镜
## 正透镜与负透镜

![[attachments/2024-01-07-13-53-14-image.png|510]]
**shape factor $\sigma$**
$$
\sigma = \dfrac{R_2+R_1}{R_2-R_1}
$$
## off axis
**离轴处会早成焦距变化，引发球差**
![[attachments/2024-01-07-13-58-52-image.png|537]]
![[attachments/2024-01-07-13-58-31-image.png]]
## CRA
CRA 的更多描述见 [[CRA/CRA]]
CRA mismatch 的分析见 [[CRA/CRA mismatch]]
成像面上主光线与光轴的夹角，包括 lens CRA 和 sensor CRA
![[attachments/2024-01-07-14-04-59-image.png]]
## 镜头材料
天然材料  玻璃  塑胶
折射、反射、吸收
材料选择可以避免**轴向色差**
### 阿贝数
在光学和透镜设计中，阿贝数又称透明材料的 V 数或常数，是材料色散（折射率随波长的变化）的近似测量值，**V 值高表示色散低**。利用高低阿贝数可以消除轴向色差
**轴向色差产生的原因**：
![[attachments/2024-01-07-14-18-30-image.png|573]]
**使用高低阿贝数玻璃消色差**
![[attachments/2024-01-07-14-19-10-image.png|456]]
### BR lens
BR镜片是采用了BR光学元件（蓝色光谱折射光学元件）的复合镜片。BR光学元件具有能大幅折射蓝色光（短波长光）的特性，可实现更理想的色像差补偿效果。参考：[佳能（中国）－ RF/EF镜头 － 技术介绍 － BR镜片 (canon.com.cn)](https://www.canon.com.cn/product/ef/info/info12.html)
![[attachments/2024-01-07-14-22-36-image.png|590]]

![[attachments/2024-01-07-14-22-45-image.png|590]]
## 制造工艺
a 粗胚 b 抽氧 c 充氮气 d 加热软化玻璃 e 加压 f 脱模
![[attachments/2024-01-07-14-12-54-image.png]]
非球面玻璃制造
![[attachments/2024-01-07-14-15-37-image.png|613]]
塑胶镜头制造
![[attachments/2024-01-07-14-16-44-image.png]]
## 光学镀膜
### 镀膜工艺比较
![[attachments/2024-01-07-16-41-41-image.png]]
**PVD**
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
## local blur
**原因**：镜头倾斜或偏移（镜片安装、音圈电机对焦时不同区域力不同、螺纹调焦没拧好）
![[attachments/2024-01-07-16-05-35-image.png|515]]

![[attachments/2024-01-07-16-05-03-image.png]]

![[attachments/2024-01-07-16-05-21-image.png]]
**MTF 表现**：曲线非中心对称
![[attachments/2024-01-07-16-06-09-image.png|551]]
## MTF
点扩散函数 PSF 见 [[PSF]]
MTF 见 [[../evaluation/MTF|MTF]]
## 光学像差
[[Optical aberration]]
畸变 [[distortion]]
# 鱼眼
![[attachments/2024-01-07-15-43-55-image.png|493]]
## 球面投影模型
![[attachments/2024-01-07-16-27-47-image.png|566]]
![[attachments/2024-01-07-16-31-38-image.png|296]]
![[attachments/2024-01-07-16-32-06-image.png|509]]
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
![[attachments/2024-01-07-16-34-24-image.png|565]]
![[attachments/2024-01-07-16-35-54-image.png|529]]

![[attachments/2024-01-07-16-35-43-image.png|536]]

![[attachments/2024-01-07-16-36-43-image.png|539]]

![[attachments/2024-01-07-16-36-18-image.png|542]]
### 鱼眼镜头的应用

监控：
![[attachments/2024-01-07-16-38-39-image.png|435]]
全景拼接：
![[attachments/2024-01-07-16-40-10-image.png]]

![[attachments/2024-01-07-16-40-03-image.png|459]]

# 其他光学配件
## Optical Density (OD滤镜/ ND filter）
$OD = \log10(I_0/I)$，其中$I_0$入射光，$I$出射光
![[attachments/2024-01-07-16-52-00-image.png]]
## IR cut filter
![[attachments/2024-01-07-16-52-24-image.png]]
白天需要IR,晚上可以不要
![[attachments/2024-01-07-16-52-59-image.png]]
## UV cut filter
用于改善紫边
![[attachments/2024-01-07-16-53-26-image.png|431]]
## 偏振
分为：线偏振、圆偏振、椭圆偏振
### 线偏振
无偏振光，经过偏振片，衰减为一半，线偏振光衰减根据夹角计算
![[attachments/2024-01-07-16-55-20-image.png]]
### 圆偏振/椭圆偏振
圆偏振可以认为是一种特殊的椭圆偏振
![[attachments/2024-01-07-16-56-56-image.png|616]]
### 偏振应用
#### 消反射光
![[attachments/2024-01-07-17-00-12-image.png]]
#### 偏振分光棱镜、双折射晶体
用两种不同偏振态的晶体组合
![[attachments/2024-01-07-16-59-02-image.png|504]]
#### 相位延迟 半波片 四分之一波片 光学低通滤波器
![[attachments/2024-01-07-17-02-15-image.png]]

应用：光学低通滤波器（消摩尔纹）
![[attachments/2024-01-07-17-03-58-image.png]]
#### LCD 显示器
利用液晶改变偏振态，调节光线强弱
![[attachments/2024-01-07-17-04-54-image.png|483]]
#### 3D glass
![[attachments/2024-01-07-17-05-49-image.png|546]]
