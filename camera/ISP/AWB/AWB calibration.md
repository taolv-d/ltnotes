---
type: note
status: done
tags:
  - camera
  - isp
  - awb
rating: 0
create: 2026-05-19
update:
---
AWB 是非常依赖标定的模块，如果AWB检测白点的算法是完美的，只要标定效果足够好，AWB表现就是OK的。
AWB 算法本质是在相机直出的色彩空间中找到哪些本该是白色的区域（这也就是AWB要标定，甚至每个sensor都要单独标定的原因）。
AWB 标定：camera 直出的R G B是一个三维的空间，我们的目的是在这个三维空间中找到不同色温光源照明时，白色物体的R G B 值。显然这是一个非常复杂的问题而且**耦合了亮度**。一种简化的方法是把三维降到二维，但是**单一二维空间不能完全分离物体颜色和光源颜色**，例如一个黄棕色物体（纸板）可能是H光下的白色物体，也可能是D50下的纸板。因此需要引入多个不同的变换来佐证（让干扰项无法在两个变换中都被判定为白点）。
 
 
### 常见标定域
 
**R/G B/G**
![[attachments/Pasted image 20260519164349.png]]

**YCrCb**
下图中 Cb Cr 都是负数，对应到YCrCb空间应该是偏绿色的部分，对应到sensor raw图直接转rgb时灰色部分的表现
![[attachments/Pasted image 20260519165006.png]]

**XY**
下图时RK ISP 标定的XY域图像。这里XY域具体变换不清楚怎么做的，但是猜测跟CIE1931中的XYZ变换有关系。
![[attachments/Pasted image 20260519170051.png|505]]

