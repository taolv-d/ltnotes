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

其他参考 

BV1za411F76U
https://www.bilibili.com/video/BV1Vd4y1e7pj
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

# 卷积
卷积早期是服务概率论的，但是卷积+傅里叶变换在信号处理上非常有用。这里单独拎出来成立一个章节。

## 积分可以看作信号与阶跃函数的卷积

TODO

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

连续的信号在进行数字化时必须进行采样，那么问题就来了，每隔多久采样一次，信号才能完整的重建出来呢？

采样过程可以看作连续信号与狄拉克梳状函数的乘积：
$$
f_s​(t)=f(t)⋅III_T​(t)=\sum_{n=-\infty}^{\infty}f(nT)\delta(t-nT)
$$
$T$为采样间隔
这里直接看时域是看不出来东西的。我们关注的时上述变换到频域后，$f(t)$的傅里叶变化$F(\omega)$ 的各个分量能否完整保留下来。利用卷积定理，时域乘积对应频域卷积，$f_s(t)$的傅里叶变换为（狄拉克梳状函数的变换在后面介绍）：
$$
F_s(\omega)=\frac{1}{T}[F(\omega)*III_{1/T}(\omega)]=\frac{1}{T}\sum_{k=-\infty}^{\infty}F(\omega-k\frac{2\pi}{T})
$$
也就是说采样后函数的频谱，是采样前函数的频谱的周期性重复，周期为$\frac{2\pi}{T}$。这里问题就来了，如果：$\frac{2\pi}{T}$ < $F(\omega)$  的频谱的宽度，那么不同周期的重复就会重叠，重叠部分就无法恢复了。

要保证不重叠，假设信号的最高频率为 $\omega_m$ ，则，他的频谱分布就是$[-\omega_m,\omega_m]$ (实信号的频谱都是正负对称的，这里可以看最后的困惑点部分)。
显然（公式是 rad/s）
$$\frac{2\pi}{T} >= 2\omega_m$$ 
也可以写成Hz 单位，就是：
$$
\frac{1}{T}>=2f_m
$$
即采样频率是信号最大频率的 2倍。(别忘了这里T是采样间隔)

### 如何采样率不足——混叠
采样率不足就会混叠，混叠就是不同频率的信号混在一起了。图像实际的表现是这样的：
1. 摩尔纹
2. 边缘锯齿
3. 虚假的纹理

### 如何恢复原信号
在频域乘理想低通滤波器



TODO

```
采样定理
负频率
混叠
理想低通滤波器

信号恢复

- **数学推导细节**（比如 `fs > 2fm` 的严格证明）
    
- **DFT泄露的本质**（为什么非整数周期会泄露，栅栏效应）
    
- **sinc插值的工程实现**（截断、加窗、过采样）
    
- **带通采样定理**（如果你的信号是射频/通信类的）
    
- **量化误差与信噪比**（ADC的实际限制）
  
  最优检测器的证明

```