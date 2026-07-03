---
type: note
status: draft
tags:
  - camera
  - optics
  - temperature
rating: 0
create: 2026-07-03
update:
---
随温度变化，成像镜头的镜片、holder等会发生一定的变化，进而改变镜头的特性。这往往需要进行温度补偿。
温度对镜头的影响主要**集中在焦距**，**畸变、光心等影响很小**。

**畸变** 这篇文章测试了投影镜头畸变随温度的变化，结论是变化在 0.1% 量级：[讨论400万像素DLP投影仪镜头的温度、电视畸变和横向色彩](https://opg.optica.org/osac/fulltext.cfm?uri=osac-2-11-3188)

**光心** 这个文章实测了光心变化，也是在 1 pixel 量级 [Image distortion of working digital camera induced by environmental temperature and camera self-heating - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0143816618311527)

**焦距随温度线性变化：**

对于焦距，通常是随温度线性变化。这几篇文章都有介绍：
[irvin_report_01.pdf](https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/10/irvin_report_01.pdf#1#1)
[被動消熱差設計介紹 | Edmund Optics](https://www.edmundoptics.com.tw/knowledge-center/application-notes/optics/an-introduction-to-passive-athermalization)

以下公式总结自上面两篇文章，镜头离焦量变化满足如下关系：
$$
\Delta f = f(\beta_{lens}-\alpha_H)\cdot\Delta T
$$
其中：
$\alpha$ 为材料的热膨胀系数，工程中认为是常数
$\beta$ 被称为**热光系数（thermo-optic coefficient）**，其定义为：
$$
\beta=\alpha_g-\frac{1}{n-1}\frac{dn}{dT}
$$
其中:
$\alpha_g$ 是热膨胀系数，也是常数
$\frac{dn}{dT}$ 折射率温度系数，对于大部分镜头材料、常温区间也认为是近似恒定值（泰勒展开后，高阶变化量很小，被公差淹没）

综上，焦距变化量可以认为是随温度线性变化。
此外，可以针对针对这个系数精心设计，可以让镜头内不同镜片受温度变化相互抵消，例如zemax可以建模补偿。这篇材料也介绍了镜头设计时的补偿：[Slide 1](http://www.sfoptique.org/medias/files/3141-rogers-athermalization-2014.pdf#2#1)

**什么时候不线性了：**
1. 对于树脂材料，温度区间到了材料接近玻璃化转变温度（Tg点），材料膨胀系数突变
2. 极端的低温（比如 -60°C以下），折射系数也不稳定

**如何测量：**
1. 利用棋盘格，可以测量到畸变、光心、焦距等随温度变化关系
2. 直接调焦，看对焦位置随温度的关系

