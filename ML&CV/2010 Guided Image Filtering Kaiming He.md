---
type: artical
status: done
tags:
  - 
rating: 0
create: 2026-07-23
update:
publish: 2010-01-01
url: https://people.csail.mit.edu/kaiming/eccv10/index.html
---
引导滤波是非常出名的保边滤波算法之一，与双边滤波齐名。


引导滤波的思想是：滤波输出是引导图像的局部线性变换.
引导滤波做类似$y=kx+b$的变换，其中$x$就是引导图，两个变换系数$k,b$ 是基于引导图跟输入图得到的（通过计算局部窗口内的均值、标准差、协方差等）

引导滤波的引导图可以是原图自己（此时类似保边滤波），也可以是其他图，例如：
1. 暗通道去雾方法中的暗通道
2. 图像融合
3. 多模态融合

# 算法推导

假设，引导图像$G$​， 输出图像$O$​，则对应图像的任意一个位置$k$，滤波器窗口为$\omega_k$，根据上述假设:
$$
O_i=a_k\cdot G_i + b_k, i \in \omega_k
$$
在保边平滑算法中，这就可以保证，滤波输出$O$的梯度和引导图$G$尽量一致。
现在给定输入图像$I$​，为了让输出$O$在**局部**内容上和$I$保持大致相同，则有最优化目标
$$
min\sum_{i\in \omega_k}(O_i-I_i)^2=min\sum_{i\in \omega_k}(a_k\cdot G_i+b_i-I_i)^2
$$
为了防止除 0（后面的简化公式有），加入一个控制变量​。
$$
min\sum_{i\in \omega_k}(a_k\cdot G_i+b_i-I_i)^2+\varepsilon \cdot a_k^2
$$
求上式的最小值，对$a_k$和$b_k$求偏导
$$
\begin{split}
\frac{\partial E}{\partial b_k}&=\sum_{i\in \omega}(2b_k+2(a_kG_i-I_i))\\
\frac{\partial E}{\partial a_k}&=\sum_{i \in \omega}(2G_i^2\cdot a_k + 2(b_i-I_i)\cdot G_i + 2\varepsilon\cdot a_k)

\end{split}
$$
另偏导为0，对$b_k$(省略sum 求和范围）:
$$
\sum b_k=\sum I_i - \sum a_kG_i
$$
求和改为计算窗口内均值：
$$
b_k=mean(I)-a_k\cdot mean(G)
$$
对于$a_k$(同样省略求和范围):
$$
\begin{split}
a_k\cdot\sum(G_i^2+\varepsilon)&=\sum(G_iI_i-G_ib_k)\\
a_k\cdot\sum(G_i^2+\varepsilon)&=\sum(G_iI_i-G_i(mean(I)-a_kmean(G)))\\
a_k\cdot\sum(G_i^2+\varepsilon-G_i\cdot mean(G))&=\sum(G_iI_i-G_imean(I))\\
a_k\cdot[\sum G_i^2+\sum\varepsilon-mean(G)\cdot\sum G_i]&=\sum (G_iI_i)-mean(I)\cdot\sum G_i\\
a_k[mean(G^2)+\varepsilon -mean(G)^2]&=mean(G\cdot I)-mean(G)\cdot mean(I)\\
\end{split}
$$
到这一步已经可以计算了，但是还可以利用协方差公式继续化简：
$$
\begin{split}
Var(X)&=E[X^2]-E[X]^2\\
Cov(X,Y)&=E[XY]-E[X]\cdot E[Y]
\end{split}
$$
带入得：
$$
a_k=\frac{Cov(G, I)}{Var(G)+\varepsilon}
$$
这里通过引导图跟输入图计算出线性变化的斜率跟偏置，接下来带入线性变换公式：
$$
O_i=a_k\cdot G_i + b_k, i \in \omega_k
$$
上面的公式看起来像是逐pixel变换，实际不是这样，实际是窗口内取平均值：
$$
O_i=\sum_{k:i\in \omega_k}(a_k\cdot G_i+b_k)
$$
这里可以简化为先逐pixel计算然后再均值滤波

## 换个角度看公式

如下式分解，可以看到输出就是 输入$I$的低频分量跟引导图的高频分量的加权求和
$$
\begin{split}
O_i&=\sum(a_k\cdot G_i+b_k)\\
&=\sum(a_k\cdot G_i + mean(I)-a_k\cdot mean(G))\\
&=\sum(mean(I)+a_k\cdot(G_i-mean(G)))
\end{split}
$$
使用原图做引导图降噪时，有 $G=I$:
- 平坦区域$G_i-mean(G)\approx0$，此时为均值滤波
- 纹理区域时$|G_i-mean(G)|> 0$，此时保留高频细节

# 参考

部分描述摘自：https://zhuanlan.zhihu.com/p/438206777
OpenCV, matlab 已经实现了相关算法