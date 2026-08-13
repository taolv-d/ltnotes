---
type: note
status: todo
tags:
  - engineering
  - fourier-transform
rating: 0
create: 2026-06-09
update: 2026-08-09
---
直观理解，傅里叶变换就是将信号分解为一些列正弦波的组合。无论是一维（如声音）还是二维（如图像），都可以通过傅里叶变换分解为一些列正弦波的加权和。
实际上，我们日常使用的 mp3 jpeg 等格式都利用傅里叶变化将高频部分压缩掉来优化文件体积。
这个交互式网站非常直观：
	https://github.com/Jezzamonn/fourier
	https://www.jezzamon.com/fourier/zh-cn.html

# 傅里叶变换是怎么想出来的

这个视频值得一看：https://www.bilibili.com/video/BV1eUHjzgEAd
**傅里叶变换，不过就是坐标分解**
傅里叶变换的其实跟很多数学上的操作本质上的思想是一致的：将一个复杂的 函数或者其他的东西分解为一些列**正交基**的线性组合。下面是几个例子：
- 二维/三维/N维 坐标系中的向量实际就是一些列正交基向量组合而成
- 泰勒展开 也是将复杂函数用n阶多项式组合
傅里叶变换就是将复杂信号用一些列 sin cos 函数组合。
**那么，为什么选就是这两个函数？**
- 首先，自然界中很多信号都跟震动有关，而震动往往都是正弦函数的
- sin cos 是性质良好的周期函数（连续可导，且导数性质很好）
- sin cos 满足正交的要求
**正交为什么重要？**
两个不正交的基其实也可以用于分解。但是，正交函数有一个很好的特性，如果你想求每个基的强度时，只需要将原始函数跟基函数**求内积**就好行。
例如：二维平面上一个向量$(m,n)$,你想计算他的横坐标，那只需要跟基向量$(1,0)$计算内基就行了。
这也正是后面计算傅里叶变换的方法，只不过傅里叶要求两个函数的内积，此时需要对两个函数采样，对应采样点相乘再求和，采样点无穷多就变成积分了。

# 傅里叶级数与傅里叶变换

先说结论：**傅里叶级数是傅里叶变换在周期信号下的离散特例**，这里区别主要在于信号是不是周期的
- 周期函数（周期为T）--傅里叶变换--> T,T/2,T/3... 等无穷个周期谐波，但是**离散的**-->傅里叶级数
- 非周期函数（周期∞）--傅里叶变换-->也是一系列周期函数，但是T无穷大，造成T/2,T/3...之间没有空隙，即**连续的**，但没有空隙

对于反变换：
- 傅里叶级数：无穷个离散频率点 **求和**
- 傅里叶变换：无穷个连续频率点 **积分**

# 前置数学知识

## 欧拉公式
$$
e^{ix}=\cos x+i \sin x
$$
$$
\begin{split}
\cos\theta&=\frac{e^{j\theta}+e^{-j\theta}}{2} \\
\sin\theta&=\frac{e^{j\theta}-e^{-j\theta}}{2j}
\end{split}
$$
其中$i,j$ 为虚数单位，特别注意 $\frac{1}{j}=-j$
这里还可以从图像上理解：欧拉公式实际是随时间变化的螺旋线，他在两个正交平面的投影就是 cos 跟 sin 函数。这也是为啥很多傅里叶变换的介绍中都会有一堆圆型画图的视频。

![[attachments/Pasted image 20260807222828.png]]

## 共轭
- 公式上共轭是把**虚部的符号反转**，即$a+jb$的共轭为$a-jb$
- 几何上，相当于把复平面上的点**以实轴镜面翻转**
- $e^{j\theta}$ 的共轭 $e^{-j\theta}$ 相当于**旋转方向反转**，模长不变，相位变号
在傅里叶变换中有：
- 实信号的共轭对称性，见[[1 Fourier transform#共轭对称性（实信号）]]
- 能量计算，帕塞瓦尔定理
- 匹配滤波器，最优检测器
- 信号的内积与正交性：$\langle x,y\rangle=\int x(t)y^*(t)dt$ (取共轭的目的是为了内积结果为：模长* 模长 * 相位差，且非负)

# 如何计算一个周期信号的傅里叶级数

开始公式前，先看看这样做有什么用。我们生活中有甚多周期信号，最常见的就是 50Hz 的市电，也常常工频干扰，很多系统都要针对50Hz/60Hz 设计专门的陷波器。
## 公式推导

傅里叶级数最初表示：
$$
x(t)=a_0+\sum^\infty_{n=1}[a_n\cos(n\omega_0t)+b_n\sin(n\omega_0t)]
$$
其中$x(t)$是周期为$T_0$的周期信号，基频$\omega_0=\frac{2\pi}{T_0}$
傅里叶级数就是想办法求 $a_n,b_n$
### 欧拉公式简化
这里我们想要得到系数 $a_n,b_n$ ，注意到欧拉公式能够简化。直接把$\cos,\sin$ 两部分替换。得到：
$$
x(t)=a_0+\sum^\infty_{n=1}\left[a_n\frac{e^{jn\omega_0t}+e^{-jn\omega_0t}}{2}+b_n\frac{e^{jn\omega_0t}-e^{-jn\omega_0t}}{2j}\right]
$$
接下来将$e^{jn\omega_0t}, e^{-jn\omega_0t}$分开整理。这里其实可以发现 n 原来是$[1,\infty)$ ,这里想到与对称扩展到$-\infty$：
- $e^{jn\omega_0t}$ 的系数变为(利用虚数的性质，参考前置)$$\frac{a_n-jb_n}{2}$$
- $e^{-jn\omega_0t}$ 的系数变为$$\frac{a_n+jb_n}{2}$$
前面提到n实际已经扩展到负数部分了，我们另：
$$c_n=\frac{a_n-jb_n}{2}, n\geq1$$
对于负频率部分
$$c_{-n}=\frac{a_n+jb_n}{2}, n\geq1$$
直流：$c_0=a_0$

最终合并为从$-\infty$到$\infty$的求和
$$
x(t)=\sum^\infty_{n=-\infty}c_ne^{jn\omega_0t}
$$
这样，原来需要两个序列$(a_n,b_n)$描述的信号，现在只需要一个复序列$c_n$就够了，而且正负频率对称、整齐，后续求系数、做系统分析都方便得多。
### 计算$c_n$
计算 $c_n$ 实际就是我们前面说的跟基频做内积来计算：
**先来看看正交是咋回事**：
我们将两个不同频率的基函数在一个周期$T_0=\frac{2\pi}{\omega_0}$内积份，有
$$
\int_{T_0}e^{jm\omega_0t}\cdot e^{-jn\omega_0t}dt
=\int_{T_0}e^{j(m-n)\omega_0t}dt
$$
- 当$m=n$时：$e^0=1$，积份结果为$T_0$
- 当$m\neq n$时，被积函数在一个完整周期内积份为0。（这里可以想象分解为sin cos 两部分，它们在一个周期内的积分都是0）

利用上面的推导，我想求系数$c_k$，只需要对原来的级数乘上$e^{-jk\omega_0t}$
$$
x(t)e^{-jk\omega_0t}=\sum^\infty_{n=-\infty}c_ne^{j(n-k)\omega_0t}
$$
接下来进行积分，其中$n\neq k$的项都为零, $n=k$ 时积分为$T_0$
$$
\int_{T_0} x(t)e^{-jk\omega_0t}dt=c_k\cdot T_0
$$

$$
c_k = \frac{1}{T_0}\int_{T_0} x(t)e^{-jk\omega_0t}dt
$$
或者更常见的写法是：
$$
c_k = \frac{1}{T_0}\int_{-T_0/2}^{T_0/2} x(t)e^{-jk\omega_0t}dt
$$
对于一般的周期信号，$c_n$ 是**复数**，包含幅度和相位信息。

- 如果信号是偶函数，$c_n$ 全是实数
- 如果信号是奇函数，$c_n$​ 全是纯虚数

### 计算$a_n,b_n$

一般不需要计算 $a_n, b_n$，$c_n$形式的表示已经足够了（除非你要画图）。前面前置知识也有推导：
$$a_n=2Re[c_n],b_n=-2Im[c_n],(n\geq1)$$

# 如何计算非周期函数的傅里叶变换

## 正变换
非周期信号是从周期信号推广来了，即非周期信号相当于周期为无穷大的周期信号。那么显然不能用离散的级数来分解频谱，（前面讨论过）非周期函数是连续频率的积分。因此需要在前面计算 $c_n$的结果上改造成连续的。

$c_n$ 的表达式如下：
$$
c_n = \frac{1}{T_0}\int_{-T_0/2}^{T_0/2} x(t)e^{-jn\omega_0t}dt
$$
定义一个新函数$X(\omega)$，他是$T_0\cdot c_n$ 在$\omega=n\omega_0$处的值：
$$
X(n\omega_0)=\int_{-T_0/2}^{T_0/2} x(t)e^{-jn\omega_0t}dt
$$
即
$$
X(\omega)=\int_{-T_0/2}^{T_0/2} x(t)e^{-j\omega t}dt
$$
接下来吧周期推广到无穷大，则：
$$
X(\omega)=\int_{-\infty}^{\infty} x(t)e^{-j\omega t}dt
$$
## 逆变换
傅里叶级数的你变换就是：
$$
x(t)=\sum^\infty_{n=-\infty}c_ne^{jn\omega_0t}
$$
在傅里叶变换中，我们定义了$X(n\omega_0)=c_n\cdot T_0$，用这个公式替换上年的$c_n$
得到：
$$
x(t)=\sum^\infty_{n=-\infty}\frac{1}{T_0}{X(n\omega_0)}e^{jn\omega_0t}
$$
周期跟频率之间的关系是：$\frac{1}{T_0}=\frac{\Delta\omega}{2\pi}$，其中$\Delta\omega=\omega_0$是谱线的间隔（当周期无穷大时，谱线间隔变为0），带入：
$$
x(t)=\frac{1}{2\pi}\sum^\infty_{n=-\infty}{X(n\omega_0)}e^{jn\omega_0t}\Delta\omega
$$
当$T_0$趋近无穷大时，$\Delta\omega$趋近无穷小，同时求和变为积分，则：
$$
x(t)=\frac{1}{2\pi}\int^\infty_{-\infty}{X(n\omega_0)}e^{jn\omega_0t}d\omega
$$

# 傅里叶变换的重要性质

| 性质   | 时域                              | 频域                                                 | 直观理解           |
| ---- | ------------------------------- | -------------------------------------------------- | -------------- |
| 线性   | $ax+by$                         | $aX+bY$                                            | 信号相加，频谱也相加     |
| 时移   | $x(t-t_0)$                      | $X(\omega)e^{-j\omega t_0}$                        |                |
| 频移   | $x(t)e^{j\omega_0 t}$           | $X(\omega-\omega_0)$                               |                |
| 尺度变换 | $x(at)$                         | $\frac{1}{\mid a \mid}X(j\frac{\omega}{a})$        | 时域扩展，频域收缩；反之亦然 |
| 对偶   | $X(t)$                          | $2\pi x(-\omega)$                                  |                |
| 时域卷积 | $x*y$                           | $X\cdot Y$                                         |                |
| 频域卷积 | $x\cdot y$                      | $\frac{1}{2\pi}X*Y$                                |                |
| 微分   | $x'(t)$                         | $j\omega X(\omega)$                                |                |
| 积分   | $\int_{-\infty}^t x(\tau)d\tau$ | $\frac{X(\omega)}{j\omega}+\pi X(0)\delta(\omega)$ |                |

**帕塞瓦尔定理（能量守恒）**
$$
\int_{-\infty}^{\infty}x(t)^2dt=\frac{1}{2\pi}\int_{-\infty}^{\infty}X(\omega)^2d\omega
$$
**实信号的共轭对称性**
若$x(t)$ 是实信号，则 $X(−\omega)=X^∗(\omega)$
频谱实部为偶函数，虚部为奇函数|
## 性质的证明

### 线性
直接利用积分运算的线性性质即可证明
对于 $ax(t)+by(t)$有
$$
\int[ax(t)+by(t)]e^{-j\omega t}dt
=a\int x(t)e^{-j\omega t}dt
+b\int y(t)e^{-j\omega t}dt
=aX(\omega)+bY(\omega)
$$
### 时移
对$x(t-t_0)$做傅里叶变换有：
$$
\int_{-\infty}^{\infty}x(t-t_0)e^{-j\omega t}dt
$$
令$\tau=t-t_0$，代入得：
$$
\int_{-\infty}^{\infty}x(\tau)e^{-j\omega (\tau+t_0)}dt=X(\omega)e^{-j\omega t_0}
$$
### 频移
对$x(t)e^{j\omega_0 t}$做傅里叶变换有：
$$
\int x(t)e^{j\omega_0 t}e^{-j\omega t}dt=\int x(t)e^{-j(\omega- \omega_0) t}dt=X(\omega-\omega_0)
$$
### 尺度变换
对$x(at)$做傅里叶变换
当$a>0$时
$$
\int x(at)e^{-j\omega t}dt
$$
令$\tau=at$，则:
$$
{\frac{1}{a}}\int x(\tau)e^{-j\omega \tau/a}d\tau=
{\frac{1}{a}}X(\frac{\omega}{a})
$$
当$a<0$时，令$a=-{\mid a \mid}$，积分限会翻转，最终结果是${\frac{1}{\mid a \mid}}X{\omega/a}$
### 对偶性质
公式，如果：
$$
x(t)\leftrightarrow X(\omega)
$$
则：
$$
X(t) \leftrightarrow 2\pi{x(-\omega)}
$$
证明：
从逆变换开始：
$$
x(t)={\frac{1}{2\pi}} {\int} {X{(\omega)} e^{j\omega t}} d\omega
$$
将$t$替换为$-t$，这里右侧积分已经变成 正变换的形式了：
$$
x(-t)={\frac{1}{2\pi}} {\int} {X{(\omega)} e^{-j\omega t}} d\omega
$$
变量名替换，用$\omega,t$ 互换：
$$
x(-\omega)={\frac{1}{2\pi}} {\int} {X{(t)} e^{-j\omega t}} dt
$$
即：$X(t) \leftrightarrow 2\pi{x(-\omega)}$
### 卷积定理
见后面卷积相关的推导
### 微分性质
公式：$x'(t) \leftrightarrow j\omega X(\omega)$
推导：
$$
{\int_{-\infty}^\infty}x'(t)e^{-j\omega t}dt=[x(t)e^{-j\omega t}]_{-\infty}^\infty
-\int_{-\infty}^\infty x(t)\cdot(-j\omega)e^{-j\omega t}dt
$$
对于真实物理信号，通常有$t\rightarrow\pm\infty, x(t)\rightarrow0$，即第一项为0，只剩下第二项，即：
$$
=j\omega \int_{-\infty}^\infty x(t)e^{-j\omega t}dt=j\omega X(\omega)
$$
### 积分性质
公式：$\int_{-\infty}^t x(\tau)d\tau \leftrightarrow \frac{X(\omega)}{j\omega}+\pi X(0)\delta(\omega)$
**利用微分定理证明（部分）**
**积分性质不能直接用微分性质逆变换来证明，原因是微分变换时，直流分量（常数项）的微分为0，但积分中不是**
这里简单介绍，不是严谨的数学推导
我们利用微分性质+直流修正的方式证明，设$y(t)=\int_{-\infty}^{t} x(\tau)d\tau$，则$y'(t)=x(t)$
两边做傅里叶变换，利用微分性质（直流修正是为了避免$\omega\neq0$时$X(\omega)$不为零引起矛盾）：
$$
X(\omega)=j\omega Y(\omega)+直流修正
$$
直流修正这里比较复杂，暂时先不计算

**利用卷积**
- 积分可以看作信号与阶跃函数的卷积，参考卷积部分
- 时域卷积，频域相乘
- 阶跃函数的傅里叶变换为$u(t)\leftrightarrow \pi\delta(\omega)+\frac{1}{j\omega}$（后面有介绍）
利用以上性质有：
$$
y(t)=\int x(t)dt=x(t)*u(t)
$$
频域有：
$$
\begin{split}
Y(\omega)&=X(\omega)\left[{\pi\delta(\omega)+\frac{1}{j\omega}}\right]\\
&=\frac{X(\omega)}{j\omega}+\pi X(0)\delta(\omega)
\end{split}
$$

### 帕塞瓦尔定理
推导：
$$
\int{\mid x(t) \mid}^2dt=\int x(t)x^*(t)dt
$$
用逆变换表示$x^*(t) = {\frac{1}{2\pi}}\int X^*(\omega)e^{-j\omega t}d\omega$，并带入得：
$$
\begin{split}
\int x(t)\left[ {\frac{1}{2\pi}}\int X^*(\omega)e^{-j\omega t}d\omega\right] dt
&= \frac{1}{2\pi}\int X^*(\omega)\left[ \int x(t)e^{-j\omega t}dt\right]d\omega \\
&=\frac{1}{2\pi}\int X^*(\omega) X(\omega) d\omega \\
&=\frac{1}{2\pi}\int{\mid X(\omega)\mid}^2 d\omega
\end{split}
$$
### 共轭对称性（实信号）
推导：
对于实信号有$x(t)=x^*(t)$，$e^{-j\omega t}$的共轭为$e^{j\omega t}$

$$
\begin{split}
X^*(\omega)&=\left[\int x(t) e^{-j\omega t}dt\right]^* \\
&=\int x^*(t) e^{-j\omega t}dt \\
&=\int x(t) e^{j\omega t}dt\\
&=X(-\omega)
\end{split}
$$

# 重要傅里叶变换对

## 冲击函数

冲击函数$\delta(t)$是在$t=0$时无穷大，在其他时刻为零，且积分为1的函数。
- 冲激响应$h(t)$完全刻画了 LTI (线性时不变)系统，$H(\omega)$就是频率响应
- 采样定理基于冲击函数推导
- 任意信号可以看作无穷个冲击函数的叠加（卷积积分）
### 变换对

| 时域$x(t)$    | 频域$X(\omega)$        |
| ----------- | -------------------- |
| $\delta(t)$ | $1$                  |
| $1$         | $2\pi\delta(\omega)$ |
### 推导
$$
X(\omega)=\int_{-\infty}^\infty\delta(t)e^{-j\omega t}dt=e^0=1
$$
对于$x(t)=1$利用傅里叶变换的对偶性可直接得到（直接积份会震荡）。（其物理意义是，时域直流分量是频域中 频率为0 处的脉冲）

## 梳状函数

- 采样定理
### 变换对
| 时域$x(t)$                                | 频域$X(\omega)$                                                               |
| --------------------------------------- | --------------------------------------------------------------------------- |
| $\sum_{n=-\infty}^\infty\delta(t-nT_s)$ | $\omega_s\sum_{k=-\infty}^\infty\delta(\omega-k\omega_s),\omega_s=2\pi/T_s$ |
### 推导

时域冲击串是周期函数，直接用傅里叶变换公式套很难计算，可以先通过傅里叶级数的形式表示，然后多傅里叶级数表示做傅里叶变换
**计算傅里叶级数**
周期$T_s$，基频率$\omega_s=2\pi/T_s$
$$
c_k = \frac{1}{T_s}\int_{-T_s/2}^{T_s/2} \delta(t)e^{-jk\omega_st}dt
=\frac{1}{T_s}
$$
则，用傅里叶级数表示$x(t)$:
$$
x(t)=\frac{1}{T_s}\sum_{k=-\infty}^\infty e^{jk\omega_st}
$$
**计算傅里叶变换**
这里利用 冲击函数的傅里叶变换+时移频移 性质有$e^{jk\omega_s t}\leftrightarrow2\pi\delta(\omega-k\omega_s)$
$$
X(\omega)=\frac{2\pi}{T_s}\sum_{k=-\infty}^\infty\delta(\omega-k\omega_s)
=\omega_s\sum_{k=-\infty}^\infty\delta(\omega-k\omega_s)
$$

## 指数衰减信号
- RC 电路的频率响应
### 变换对
| 时域$x(t)$            | 频域$X(\omega)$             |
| ------------------- | ------------------------- |
| $e^{-at}u(t),a>0$   | $\frac{1}{a+j\omega}$     |
| $e^{-a\mid t \mid}$ | $\frac{2a}{a^2+\omega^2}$ |

### 推导
**对于单边指数**
$$
\begin{split}
X(\omega)&=\int_0^\infty e^{-at} e^{-jwt}dt\\
&=\int_0^\infty e^{-(a+jw)t}dt\\
&=\frac{1}{a+j\omega}
\end{split}
$$
**对于双边指数**
$$
\begin{split}
X(\omega)&=\int_{-\infty}^0 e^{at} e^{-jwt}dt + \int_0^{\infty}e^{-at} e^{-jwt}dt\\
&=\frac{1}{a-j\omega} + \frac{1}{a+j\omega}\\
&=\frac{2a}{a^2+\omega^2}
\end{split}
$$
## 高斯函数
- **唯一自对偶函数**：时域和频域都是高斯形，具有最优的时频聚集性
### 变换对
| 时域$x(t)$    | 频域$X(\omega)$                            |
| ----------- | ---------------------------------------- |
| $e^{-at^2}$ | $\sqrt{\frac{\pi}{a}}e^{-\omega^2/(4a)}$ |
### 推导
$$
X(\omega)=\int_{-\infty}^{\infty}e^{-at^2} e^{-j\omega t}dt
$$
这里需要操作下指数，变成完全平方形式（第一部分用高斯积分公式，第二部分是常数）：
$$
-at^2-j\omega t=-a(t+\frac{j\omega}{2a})^2-\frac{\omega^2}{4a}
$$
利用高斯积分公式：$\int_{-\infty}^\infty e^{-a(t+b)^2}dt=\sqrt{\frac{\pi}{a}}$
$$
X(\omega)=\sqrt{\frac{\pi}{a}}e^{-\omega^2/(4a)}
$$

## 阶跃函数
- 开关、控制系统常用
- 积分可以看作信号与阶跃函数的卷积
### 变换对

| 时域$x(t)$ | 频域$X(\omega)$                         |
| -------- | ------------------------------------- |
| $u(t)$   | $\pi\delta(\omega)+\frac{1}{j\omega}$ |
### 推导
这里将阶跃函数分解为 直流 + 符号函数，即：
$$
u(t)=\frac{1}{2}+\frac{1}{2}sgn(t)
$$
直流部分已经在前面单位冲击函数部分推导了:$1\leftrightarrow2\pi\delta(\omega)$。
符号函数推导较复杂，直接给出结果：$sgn(t)=\frac{2}{j\omega}$
则：
$$
X(\omega)=\pi\delta(t)+\frac{1}{j\omega} 
$$
## 矩形脉冲
矩形脉冲是非常重要的函数，理想采用矩形脉冲近似。它也解释了为什么无法实现理想低通滤波器、光学的衍射强度分布的规律
### 变换对

| 时域$x(t)$                                               | 频域$X(\omega)$                             |
| ------------------------------------------------------ | ----------------------------------------- |
| $x(t)=1,\mid t\mid, \mid t \mid \leq \tau/2$<br>其他情况为0 | $\tau\cdot sinc(\frac{\omega\tau}{2\pi})$ |
### 推导
这里利用了$\int e^{-j\omega t}dt=\frac{e^{-j\omega t}}{-j\omega}$
$$
\begin{split}
X(\omega)&=\int_{-\tau/2}^{\tau/2} e^{-j\omega t}dt\\
&=\frac{e^{-j\omega\tau/2}-e^{j\omega\tau/2}}{-j\omega}(带入积分上下限)\\
&=\frac{-2j\sin(\omega\tau/2)}{-j\omega}(欧拉公式)\\
\end{split}
$$
转换为标准 sinc 形式：$sinc(z)=\frac{\sin(\pi z)}{\pi z}$
令$z=\frac{\omega\tau}{2\pi}$
$$
\frac{2\sin(\omega\tau/2)}{\omega}=\tau\frac{\sin(\omega\tau/2)}{\omega \tau / 2}=\tau\frac{\sin(\pi\frac{\omega\tau}{2\pi})}{\pi\frac{\omega\tau}{2\pi}}=\tau\cdot sinc(\frac{\omega\tau}{2\pi})
$$
## 正余弦信号
### 变换对

| 时域$x(t)$           | 频域$X(\omega)$                                            |
| ------------------ | -------------------------------------------------------- |
| $\cos(\omega_0 t)$ | $\pi[\delta(\omega-\omega_0)+\delta(\omega+\omega_0)]$   |
| $\sin(\omega_0t)$  | $-j\pi[\delta(\omega-\omega_0)-\delta(\omega+\omega_0)]$ |
### 推导
这里利用 冲击函数的傅里叶变换+时移频移 性质有$e^{jk\omega_s t}\leftrightarrow2\pi\delta(\omega-k\omega_s)$，以及欧拉公式。这里就不展开了。

# 傅里叶变换成立的条件

**狄利克雷条件**是判断周期信号是否存在傅里叶级数或能展开傅里叶级数的**充分**条件（数学上没有统一的 充分必要 条件）
##  狄利克雷条件
1. **有限间断点**：信号在一个周期内必须连续，或者仅有有限个第一类间断点（即左右极限存在且有限）
2. **有限极值**：信号在一个周期内的极大值和极小值数量有限 
3. **绝对可积**：信号在周期内的**绝对值积分**有限，即信号在数学上是可积的 。  

简单理解，参考下式：
傅里叶变换定义为：$X(\omega)=\int x(t)e^{-j\omega t}dt$，且 $\mid e^{-j\omega t}\mid=1$，因此有：
$$
\mid X(\omega)\mid \leq \int{\mid x(t)\mid}\cdot{\mid e^{-j\omega t} \mid} dt=\int{\mid x(t) \mid}dt
$$
显然，必须要$x(t)$ 绝对值可积，$X(\omega)$才存在

## 不满足怎么办

很多函数，尤其是一些重要函数是不满足这个条件的：
- 冲击函数：不是普通函数
- 常量：面积无限
- 正余弦信号：不衰减
- 阶跃信号：面积无限

**应对办法：**
1. 物理信号 是有界、且能量有限的，可以近似用傅里叶变换分析
2. 正弦波、阶跃、方波等不绝对可积但**功率有限**的信号，我们使用**冲激函数**来表示其频谱。（参考傅里叶变换对）
3. 拉普拉斯变换替代
4. 计算机处理，用离散傅里叶变换替代（只要有限长，就可以用DFT近似）

# 卷积
卷积早期是服务概率论的，但是卷积+傅里叶变换在信号处理上非常有用。这里单独拎出来成立一个章节。

## 积分可以看作信号与阶跃函数的卷积
先看公式，然后再直观理解
设信号为$x(t)$，阶跃函数$u(t)$的定义如下：
$$
u(t)=\begin{cases}
1,&t\geq 0\\
0,&t<0
\end{cases}
$$
根据卷积的定义：
$$
\begin{split}
(x*u)(t)&=\int_{-\infty}^{\infty}x(\tau)\cdot u(t-\tau)d\tau \\
&=\int_{-\infty}^t x(\tau)d\tau
\end{split}
$$
这一波变换，就从卷积变到$(-\infty, t]$ 的积分了。因此这也是这个变换的限制。
**直观描述：**
- 卷积运算描述了一个“信号”与一个“系统响应”之间的叠加过程。
- 积分运算，可以拆解为信号与**单位阶跃函数**的卷积

**这样做的好处：**
- 简化分析难度，利用卷积的性质，在频域分析
- 利用傅里叶变换加速

## 卷积的基本性质
傅里叶变换中卷积的性质可以简化计算：要计算时域卷积，可以先FFT变换到频域，频域相乘，在iFFT 变换回时域。这一波操做就将 $O(n^2)$ 复杂度的计算降低到$O(nlog(n))$

### 时域卷积性质证明
卷积定义：
$$
(x*y)(t)={\int_{-\infty}^{\infty}x(\tau)y(t-\tau)}d\tau
$$
傅里叶变为为
$$
\begin{split}
\int_{-\infty}^\infty
\left[ {\int_{-\infty}^{\infty}x(\tau)y(t-\tau)}d\tau\right]
e^{-j\omega t}
dt &=
\int_{-\infty}^\infty x(\tau)
\left[ 
\int_{-\infty}^{\infty}
y(t-\tau) e^{-j\omega t}
dt
\right]
d\tau
\end{split}
$$
令$u=t-\tau$，则$t=u+\tau,dt=du$:
$$
\begin{split}
\int_{-\infty}^\infty x(\tau)
\left[ 
\int_{-\infty}^{\infty}
y(u) e^{-j\omega (u+\tau)}
du
\right]
d\tau&=\int_{-\infty}^\infty x(\tau)
e^{-j\omega \tau}
\left[ 
\int_{-\infty}^{\infty}
y(u) e^{-j\omega u}
du
\right]
d\tau \\
&=\int_{-\infty}^\infty x(\tau)
e^{-j\omega \tau} Y(\omega)
d\tau\\
&=Y(\omega)
\int_{-\infty}^\infty x(\tau) e^{-j\omega \tau}d\tau \\
&=X(\omega)Y(\omega)
\end{split}
$$
### 频域卷积定理
直接从定义出发，计算$x(t)y(t)$的傅里叶变换为：
$$
\int_{-\infty}^{\infty}x(t)y(t)e^{-j\omega t}dt
$$
将$x(t)$用逆变换形式表示：$x(t)=\frac{1}{2\pi}\int_{-\infty}^\infty X(u)e^{jut}du$
带入傅里叶变换公式有：
$$
\begin{split}
\int_{-\infty}^{\infty}x(t)y(t)e^{-j\omega t}dt
&=\int_{-\infty}^{\infty} 
{\left[\frac{1}{2\pi}\int_{-\infty}^\infty X(u)e^{jut}du \right]}
y(t)e^{-j\omega t}dt\\
&=={\frac{1}{2\pi}}
\int_{-\infty}^{\infty} X(u)
{\left[\int_{-\infty}^\infty y(t)e^{-j(\omega-u) t} dt\right]}
du\\
&={\frac{1}{2\pi}}
\int_{-\infty}^{\infty} X(u)Y(\omega -u)du\\
&={\frac{1}{2\pi}}(X*Y)(\omega)
\end{split}
$$

# 采样定理

TODO:这里信号用f(t)表示，跟其他用x(t)表示不统一
## 奈奎斯特采样定理
连续的信号在进行数字化时必须进行采样，那么问题就来了，每隔多久采样一次，信号才能完整的重建出来呢？

**时域上**：
采样过程可以看作连续信号与狄拉克梳状函数的乘积：
$$
f_s​(t)=f(t)⋅III_{T_s}​(t)=\sum_{n=-\infty}^{\infty}f(nT)\delta(t-nT_s)
$$
$T_s$为采样间隔，$f_s(t)$为采样结果

**频域上**：
我们关心各个频率分量能否完整保留（完整保留就可恢复）
利用卷积定理，时域乘积对应频域卷积，$f_s(t)$的傅里叶变换为：
$$
\begin{split}
F_s(\omega)
&=\frac{1}{2\pi}[F(\omega)*III_{1/T_s}(\omega)]\\
&=\frac{1}{2\pi} F(\omega)*\left[\omega_s\sum_{k=-\infty}^{\infty}\delta(\omega-k\omega_s) \right]\\
&=\frac{1}{T_s}\left[\sum_{k=-\infty}^{\infty}F(\omega-k\omega_s) \right]
\end{split}
$$
其中$\omega_s = 2\pi / T_s$，即最高采样间隔
第二行带入了 梳妆函数的傅里叶变换。第三行利用冲激函数的定义（$\delta(0)=1,other=0$）

也就是说采样后函数的频谱，是采样前函数的**频谱的周期性重复**，周期为$\frac{2\pi}{T}$（采样周期算出来的）。这里问题就来了，如果：
**$\frac{2\pi}{T}$ < $F(\omega)$  的频谱的宽度，那么不同周期的重复就会重叠，重叠部分就无法恢复了**。

要保证不重叠，假设信号的最高频率为 $\omega_m$ ，则，他的频谱分布就是$[-\omega_m,\omega_m]$ 。
显然（单位是 rad/s）
$$\frac{2\pi}{T_s} >= 2\omega_m$$ 
也可以写成Hz 单位，就是：
$$
\frac{1}{T_s}>=2f_m
$$
即采样频率是信号最大频率的 2倍。(别忘了这里T是采样间隔)

**上面的描述有点绕**
这里有两个频率：
1. $\omega_s = 2\pi / T_s$，他是采样信号（梳妆函数）的频率
2. $w_m$ 他是原始信号的频率
公式推导发现 采样后函数$f_s(t)$的傅里叶变换$F_s(\omega)$ 是采样前函数$f(t)$ 的傅里叶变换$F(\omega)$ 的周期性重复。
那么，这里为了能分开频谱，就不能让重复的频谱相互重叠。也就是看 重复的周期 是不是比频谱的范围大。由此得到：采样信号是 原始信号中最高频率2倍的关系。

**工程上不能用临界2倍，需要留有余量：2.5倍 或者更高**

## 如何采样率不足——混叠
采样率不足就会混叠，混叠就是不同频率的信号混在一起了。图像实际的表现是这样的：
1. 摩尔纹
2. 边缘锯齿
3. 虚假的纹理

## 信号重建

### 完美恢复不可能
**频域上**：
我们只需要一个矩形脉冲（理想低通滤波器）截取一个完整周期就行：
$$
F(\omega)=F_s(\omega)\cdot H(\omega)
$$
$$
H(\omega)=\begin{cases}
T_s,&{\mid \omega \mid}\leq \omega_c\\
0,&{\mid \omega \mid}> \omega_c
\end{cases}
$$
其中 $\omega_m \leq \omega_c\leq\omega_s-\omega_m$
 这在数学上是能完美恢复原始信号的，但是物理世界是不行的。具体看时域公式。

**时域上**：
- 采样信号为：$f_s(t)=\sum_{n=-\infty}^{\infty}f(nT_s)\delta(t-nT_s)$
- 理想低通滤波器的逆变换（参考矩形脉冲那里的介绍，这里不展开推导了）：$h(t)=sinc(t/T_s)$
利用卷积的性质有：
$$
\begin{split}
f(t)&=f_s(t)*h(t)\\
&=\int_{-\infty}^{\infty}
\left[\sum_{n=-\infty}^{\infty}f(nT_s)\delta(\tau-nT_s)\right]
sinc(\frac{t-\tau}{T_s}) d\tau\\
&=\sum_{n=-\infty}^{\infty}f(nT_s)\int_{-\infty}^{\infty}\delta(\tau-nT_s)
sinc(\frac{t-\tau}{T_s}) d\tau\\
&=\sum_{n=-\infty}^{\infty}f(nT_s)sinc(\frac{t-nT_s}{T_s})
\end{split}
$$
也就是：把每个采样点用一个$sinc$脉冲扩散到整个时间轴上，然后叠加。
到这里，可以看到要完美恢复，需要在时域无限长的范围内求和。这里有两个主要限制：
1. 无限长采样是不可能的
2. 需要未来的采样才能恢复（非因果），实时系统不行
3. 任何滤波器都不是完美矩形，都有过渡带

### 工程上怎么做

1. 零阶保持+数字域滤波器。大部分DAC实现：在每个采样点之后，输出电压保持恒定，直到下一个采样点到来。就像“台阶波”。
2. 线性差值（一阶保持）：用直线连接两个采样点，平滑波形（高频损失比零阶保持高，适用于高频不敏感场景）
3. 高阶重建：先用数字域做**高精度插值**（比如用 FIR 滤波器实现截断的 sinc 插值），再用**高性能模拟低通滤波器**平滑输出（效果好，但成本高）

几个值得考虑的点：
- 过采样能够提高重建质量
- 数字域插值、滤波更灵活
- 延迟换质量

## 最优检测器

最优检测器就是相关运算。
下面是数学推导：
**问题**：
一个已知的实信号$s(t)$，被加性高斯白噪声$n(t)$(双边功率谱密度为$N_0/2$)污染后：
$$
r(t)=s(t)+n(t),0≤t≤T
$$
设计一个**线性时不变（LTI）滤波器** $h(t)$，使得在 $t=t_0$ 时刻的**输出信噪比（SNR）**最大.
### 证明(求滤波器的时域表达式)
在 $t=t_0$​ 时刻
$$
y(t_0)=r(t)*h(t)=\int_{0}^{T}r(\tau)h(t_0-\tau)d\tau
$$
**信号分量**
$$
y_s(t_0)=∫_0^Ts(τ)h(t_0−τ)dτ
$$
**噪声分量**
$$
y_n(t_0)=∫_0^T n(τ)h(t_0−τ)dτ
$$
噪声是零均值的，所以输出噪声的方差（功率）为：
$$
\sigma_n^2​=E\{y_n^2​(t_0​)\}=frac{N_0}{2}​​∫_0^T​∣h(t_0​−τ)∣^2dτ
$$
（这是线性系统对白噪声的响应公式：$输出噪声功率谱密度 = 输入噪声功率谱密度 × ∣H(ω)∣^2$，再积分得到总功率。）
**信噪比**
$$
{SNR}_{out}​=\frac{​∣y_s​(t_0​)∣^2}{σ_n^2}​
=\frac{\mid \int_0^Ts(\tau)h(t_0-\tau)d\tau\mid^2}
{\frac{N_0}{2}\int_0^T\mid h(t_0-\tau) \mid^2d\tau}​
$$
令 $g(τ)=h(t0−τ)$，则 SNR 表达式变为：
$$
{SNR}_{out}​=\frac{​∣y_s​(t_0​)∣^2}{σ_n^2}​
=\frac{\mid \int_0^Ts(\tau)g(\tau)d\tau\mid^2}
{\frac{N_0}{2}\int_0^T\mid g(\tau) \mid^2d\tau}​
$$
根据**柯西-施瓦茨（Cauchy-Schwarz）不等式**：
$$
{\left|\int_0^Ts(\tau)g(\tau)d\tau \right|^2}
\leq 
\left( \int_0^T|s(\tau)|^2d\tau \right)
\cdot
\left( \int_0^T|g(\tau)|^2d\tau \right)
$$
等号成立的条件是：$g(τ)=k⋅s(τ)$，即：
$$
h(t_0-\tau)=k\cdot s(\tau)
$$
变换下变量名，令$t = t_0-\tau$,并令$k=1$(通常)；令$t_0=T$，即信号结束时判断。上式变换为：
$$
h(t)=s(T-t)
$$
这个公式就是相关运算，用自己做模版去扫描接收的波形
### 频域表达式
对 $h(t)=s(T−t)$ 做傅里叶变换：
$$
H(ω)=∫_{−∞}^∞ s(T−t) e^{−jωt}dt
$$
令 $τ=T−t$，则：
$$
\begin{split}
H(ω)&=∫_{−∞}^∞ s(τ)e^{−jω(T−τ)}⋅(−dτ) \\
&=e^{−jωT} ∫_{−∞}^∞ s(τ)e^{jωτ}dτ \\
&=S(-\omega)e^{−jωT}\\
&=S^*(\omega)e^{−jωT}
\end{split}
$$
其中，$S(−ω)=S∗(ω)$对实信号成立。

匹配滤波器在频域上**补偿了信号的相位**，使得所有频率分量在某一时刻同相叠加，达到最大峰值。


```
- **DFT泄露的本质**（为什么非整数周期会泄露，栅栏效应） 
- **带通采样定理**（如果你的信号是射频/通信类的）
- **量化误差与信噪比**（ADC的实际限制）
```

# 离散傅里叶变换

计算机、信号处理中，只能处理**有限长的离散序列**，因此需要离散傅里叶变换（DFT）

## DFT 的定义
对于长度为 $N$ 的离散序列 $x[0],x[1],…,x[N−1]$，其 DFT 定义为：
$$
X[k]=\sum_{n=0}^{N−1} x[n]⋅e^{−j\frac{2π}{N}kn},k=0,1,…,N−1
$$
逆变换（IDFT）为：
$$
x[n]=\frac{1}{N}\sum_{k=0}^{N−1}X[k]⋅e^{j\frac{2π}{N}kn},n=0,1,…,N−1
$$
## 连续到离散

从连读到离散的主要核心就是：采样+截断。采样是连续到离散。截断是无穷到有限
### **时域离散化**
我们用梳状函数采样得到离散函数：
采样周期为 $T_s$​，采样频率 $f_s=1/Ts$​，角频率 $ω_s=2πf_s=2π/T_s$​。
采样信号的频谱（推导见采样定理）：
$$
X_s(\omega)=\frac{1}{T_s}\sum_{k=-\infty}^{\infty}X(\omega-k\omega_s)
$$
时域的离散时间序列定义为：$x[n]=x(nT_s​),n∈Z$
对$x[n]$ 做傅里叶变换，就是离散时间傅里叶变换（DTFT）。DTFT 用的不多，这里直接给出他的公式（推导也是从FT过来的）
$$
X(e^{j\Omega})=\sum_{n=-\infty}^{\infty}x[n]e^{-j\Omega n}
$$
其中，$\Omega$ 是归一化数字角频率，单位是弧度，与模拟角频率 $ω$ 的关系为：$Ω=ωTs$
显然：
$$
X(e^{j\Omega})=X_s(\frac{\Omega}{T_s})
$$
**额外多解释下**
这里回看下采样定理的推导，梳状函数可以看成在时刻 $t=nT_s$的采样，就是离散时间信号$x[n]$。因此采样信号的傅里叶变换跟离散时间信号的傅里叶变换等价（令$\Omega=\omega T_s$）:
$$
\begin{split}
X_s​(ω)&=\sum_n​ x(t)\delta(t-nT_s)e^{−jωnT_s}\\
&=\sum_n​ x[n]e^{−jωnT_s}\\
&=\sum_n​ x[n]e^{−j\Omega n}\\
&=X(e^{j\Omega})
\end{split}​
$$
### 频域采样
$$
X(e^{j\Omega})=\sum_{n=-\infty}^{\infty}x[n]e^{-j\Omega n}
$$
这个变换$\Omega$是连续的，因此整体还是连续的。需要对频谱采样:
在一个周期 $[0,2π)$ 内等间隔取 $N$ 个点：
$$Ω_k=\frac{2πk}{N},k=0,1,…,N−1$$
对应的模拟角频率为：
$$
ω_k=\frac{Ω_k}{T_s}=\frac{2πk}{NT_s}​
$$
变换变为：
$$
X[k]=X(e^{j\Omega_k})=\sum_{n=-\infty}^{\infty}x[n]e^{-j\frac{2\pi}{N} kn}
$$
### 截断
$X[k]$ 需要无限长求和，显然是不能实现的，需要截断为0到$N-1$
$$
X[k]=\sum_{n=0}^{N-1}x[n]e^{-j\frac{2\pi}{N} kn},k=0,1,...,N-1
$$
至此DFT的变换公式就出来了

**逆变换**：逆变换的思想是一致的，但是推导更复杂一些。这里先不展开了

## 精度损失
上面推导中：用到采样、截断等，显然会损失精度，这是理论层面的固有缺陷。因此DF T 是精度换可计算性的工具
**但是：**
**DFT 和 IDFT 是一对精确的数学变换对，互相转换没有任何精度损失。**
只要信号本身**就是周期的**，并且你截取的长度**恰好等于整数个周期**，那么“截断”就不会引起频谱泄露，“频域采样”也能正好对准谱线。此时，DFT 的计算结果就是这个周期信号的精确定义。
### 什么时候不损失
对于一个**严格周期**的信号，且我们**恰好截取了整数个周期**，那么离散傅里叶变换（DFT）在这个特定任务上是**没有精度损失的**

1. **信号是严格周期的**：周期为 $T_0$。
2. **满足采样定律**：采样频率 $f_s$ 远大于信号最高频率 $f_m$​。
3. **截取的是整数个周期**：截取长度 $T=L⋅T_0$（$L$ 为正整数），采样点数 $N=T/T_s​$。

在这种情况下，**“截断”和“频域采样”这两个导致误差的操作，恰好与信号的天然属性完美匹配**：
- **截断（无频谱泄露）**：因为截取的长度正好是周期的整数倍，截断后的片段首尾相接，完美还原了信号的周期性。在频域上，原来连续周期的频谱，被自然地采样成了离散的谱线，没有能量“泄露”到相邻频点。
- **频域采样（无栅栏效应）**：DFT 的频域采样点 $Ω_k=2πk/N$，正好落在了信号基频 $Ω_0=2π/N_0$​（$N_0$​ 是一个周期内的点数）的整数倍上。这意味着，DFT 的“栅栏”恰好对准了所有有能量的谱线，没有漏掉任何峰值。

## 二维离散傅里叶变换

二维离散傅里叶变换（2D-DFT）的定义为:
$$
F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)\cdot e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}
$$
其中
$$
e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}=
e^{-j2\pi\frac{ux}{M}}\cdot e^{-j2\pi\frac{vy}{N}}
$$
代回去，就是：
$$
F(u,v)=\sum_{x=0}^{M-1} \left[ \sum_{y=0}^{N-1}f(x,y)\cdot e^{-j2\pi(\frac{ux}{M})}\right]\cdot e^{-j2\pi\frac{vy}{N}}
$$
即，分为两个一个傅里叶变换（先行后列，或者先列后行），也就是**可分离性**

### 频谱图特征
1. 布局：中心是低频（亮度），向外是高频（细节/边缘/噪声）
2. 方向性：频谱上的“亮点”对应图像中的“条纹”
	-   **水平方向的亮点**：表示图像中有**垂直的条纹或边缘**（因为垂直边缘在水平方向上变化最快）。
	-   **垂直方向的亮点**：表示图像中有**水平的条纹或边缘**。
	-   **对角线方向的亮点**：表示图像中有**相应角度的斜向纹理**。
	-   **一圈圈的亮环**：表示图像中有各向同性的纹理，比如同心圆或随机噪声。
3. 严格中心对称（幅度谱）
4.  幅度与相位：一个管“亮度”，一个管“内容”
	-   **幅度谱（亮度）**：决定了图像的**能量分布**和**灰度对比度**。比如，边缘多、纹理丰富的图片，其高频分量（远离中心）的亮度就会更亮。
	-   **相位谱（隐含信息）**：记录了每个频率分量的**位置信息**。这才是决定图像“内容”（比如这是一只猫还是一栋楼）的**关键**。
	- 把两张不同图片的幅度谱和相位谱互换，进行逆变换。结果会显示，**逆变换后的图像内容几乎完全由相位谱决定**

## 快速傅里叶变换

快速傅里叶变换通过**分治**将复杂度从$O(N^2)$降到$O(N\log N)$
先看DFT的公式：
$$
X[k]=\sum_{n=0}^{N−1} x[n]⋅e^{−j\frac{2π}{N}kn},k=0,1,…,N−1
$$
求和公式中有两个独立变量$n、k$，因此需要的乘法次数为：$n*k=N^2$
### 加速的原理
FFT 主要利用了$e^{−j\frac{2π}{N}kn}$ 的性质。为简化表示，令$W_N=e^{−j\frac{2π}{N}kn}$：

这里还有两个要用的结果（利用欧拉公式应该能计算出来，这里不展开）：
$e^{-j2\pi}=1$, $W_N^{N/2}=e^{-j\pi}=-1$

**周期性**
$$
\begin{split}
W_N^{k+N}&=e^{−j\frac{2π}{N}(k+N)n} \\
&=e^{−j\frac{2π}{N}kn}\cdot e^{−j2πn}\\
&=e^{−j\frac{2π}{N}kn}\\
&=W_N^k
\end{split}
$$
**对称性**
$$
\begin{split}
W_N^{k+N/2}&=e^{−j\frac{2π}{N}(k+N/2)n} \\
&=e^{−j\frac{2π}{N}kn}\cdot e^{−jπn}\\
&=-e^{−j\frac{2π}{N}kn}\\
&=-W_N^k
\end{split}
$$
这里**对称性**说明：**在 NN 个旋转因子中，真正独立的只有 N/2N/2 个，剩下的一半只是它们的“相反数”。**
利用相反数，就可以少算一半
### 公式推导

$$
X[k]=\sum_{n=0}^{N−1} x[n]\cdot W_N^{kn} ,k=0,1,…,N−1
$$
先分为奇偶两部分：
这里有：$W_N^{2kn}=e^{-j\frac{2\pi}{N}2kn}=e^{-j\frac{2\pi}{N/2}kn}=W_{N/2}^{kn}$  
注意$k=0,1,2,...,N/2-1$
$$
\begin{split}
X[k]&=\sum_{n=0}^{N/2−1}x[2n]W_N^{k(2n)}+ \sum_{n=0}^{N/2−1}x[2n+1]W_N^{k(2n+1)}\\
&=\sum_{n=0}^{N/2−1}x[2n]W_{N/2}^{kn}+ W_N^k\sum_{n=0}^{N/2−1}x[2n+1]W_{N/2}^{kn}\\
&=E[k]+W_N^kO[k]
\end{split}
$$
这里在利用周期性（注意上式第二行变换，周期已经变为$N/2$, 但序列长度是$N$）: 
利用对称性（主要是$O[k]$前的$W_N^k$变为负号）,跟周期性($E[k],O[k]$不变)，写出$X[k]$的前后两段分别为:
下面推导中$k=0,1,...,N/2-1$
**前半段**：$$X[k]=E[k]+W_N^kO[k]$$
**后半段**:  
$$
\begin{split}
X[k+N/2]&=E[k+N/2]+W_N^{k+N/2}O[k+N/2]\\
&=E[k]+W_N^{k+N/2}O[k](周期性)\\
&=E[k]-W_N^{k}O[k](对称性)
\end{split}
$$
这样一分解，$E[k],O[k]$ 只需要算一半就行了，另外一半直接边个符号重新相加。（即$N$ 个加法运算，$N/2$个乘法运算）

接下来，对于每个子序列（$E[k],O[k]$）也能看作一个采样率不一样的数据，继续套这个分解过程，递归计算，知道只剩一个点了。（这个跟排序算法的分治思想一致）

这里的FFT 并非最快，还有其他改进，这里就不展开了。

**8点FFT 分解示意图**
![[attachments/Pasted image 20260812234547.png]]
# FIR IIR 滤波器