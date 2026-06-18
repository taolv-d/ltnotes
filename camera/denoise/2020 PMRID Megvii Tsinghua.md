---
type: artical
status: done
tags:
  - camera
  - denoise
rating: 0
create: 2026-05-13
publish: 2020
url: 
update:
---

# k-sigma 变换降低降噪难度

- **难点**：ISO 100 和 ISO 6400 的噪声完全不是一个量级。如果直接喂给一个网络，网络会很困惑，难以收敛（或者需要一个大网络才能处理）。

- **做法**：设计了一个数学变换 $f(x)=\frac{x}{k}+\frac{\sigma^2}{k^2}$。

- **效果**：经过这个变换后，**无论原始ISO是多少，噪声的分布都会被映射到一个统一、稳定的空间**(即方差`==`均值)。在这个空间里，ISO 100和ISO 6400的图像看起来“噪声水平差不多”。


**两个系数如何标定**
	论文中提到的噪声模型标定 现在已经比较常见，很多商用ISP已经支持（例如地平线地瓜x5 ISP 中已经集成vst变换模块）

基于**泊松-高斯噪声模型**来描述RAW图像的噪声：
- **散粒噪声**：与进光量有关，光越强，噪声波动越大。
- **读出差分噪声**：与传感器电路有关，和进光量无关，但和ISO（增益）强相关。

- **测量方法**： 对着灰度卡连拍64张，算出每个像素位置的**均值**（当作真值）和**方差**（当作噪声强度）。通过线性回归，就能拟合出不同ISO下的噪声参数 $k$ 和 $\sigma^2$。

**对比 VST 变换**
- k-Sigma 是线性变换，VST 是非线性变换[[VST]]，显然k-Sigma 变换会更简单
- 对比VST 方差 ~= 1

**k-Sigma 的局限**
1.  需要标定
2. 将泊松分布近似为高斯，但这再极暗光条件不成立，泊松分布不是左右对称的，VST 变换能够更好处理这种情况
3. k-Sigma 仍然没有把不同增益拉到完全相同的水平下，即不同增益变换后的方差仍然不是完全相同的，这对网络的要求更高。相反VST让方差变为1，网络更容易学去噪，YOND中用VST变换+简单网络实现了更好的泛化性[[2025 YOND]]

# pmrid 网络 极度轻量化

本论文的核心是 k-sigma 变换，因此使用的是非常轻量的网络，目的就是证明他的变换缺失降低了问题复杂度。
PMRID 的网络结构就是“RAW 4通道输入 + ISO噪声归一化 + 轻量残差U-Net + 深度可分离卷积 + 全局残差输出”。整个网络没有 BatchNorm、没有 attention、没有 transformer。甚至将UNet中的concat 操作替换成直接相加

- **基础框架**：4层编码器+4层解码器的 **U-Net** 结构。
- **计算瘦身**：大量使用**深度可分离卷积**[[../../machine learning/nn积木/DSC 深度可分离卷积|DSC 深度可分离卷积]]，只在输入输出层用普通卷积。

数据在网络中大概按如下方式流动
```
输入->按bayer格式拆分为四通道，即[1,1,W,H]->[1,4,W/2,H/2]
UNet 四个下采样 + middle + 四个上采样
3*3 卷积到 4 通道，输出 4 通道噪声修正量 out
全局残差学习：pred = input + out
pixel shuffle 还原为raw格式
```

# k-Sigma变换推导

### 噪声模型
$$ x = g(\alpha u + n_d) + n_r $$
其中：
*   $g$：模拟增益（与ISO相关）
*   $\alpha$：光电转换的量子效率
*   $u$：实际接收到的光子数，$u \sim P(u^*)$ （$u^*$ 为期望光子数，$P$ 为泊松分布）
*   $n_d$：暗电流噪声，$n_d \sim \mathcal{N}(0, \sigma_d^2)$
*   $n_r$：读出噪声，$n_r \sim \mathcal{N}(0, \sigma_r^2)$

为了简化表达，论文引入了两个变量 $k$ 和 $\sigma^2$：
*   令 $k = g\alpha$
*   令 $\sigma^2 = g^2\sigma_d^2 + \sigma_r^2$

此时，原始像素值 $x$ 服从一个高斯-泊松联合分布（热噪声+散粒噪声）：
$$ x \sim kP\left(\frac{x^*}{k}\right) + \mathcal{N}(0, \sigma^2) $$

由此可得 $x$ 的均值与方差特性：
$$ E(x) = x^* $$
$$ Var(x) = kx^* + \sigma^2 $$

此时噪声的方差 $Var(x)$ 依赖于信号 $x^*$ 和传感器增益 $g$。不同增益下，$k$ 和 $\sigma$ 都会变化，导致噪声分布差异巨大。

---
这里的推导利用了全方差公式 ，（两条线之间为推导全过程）
$$ Var(X) = E[Var(X|Y)] + Var(E[X|Y]) $$
对于噪声模型公式，令：
- $Y = u$ （实际光子数，服从泊松分布）
- $X = x$ （原始像素值）

用全方差公式：
$$ Var(x) = E[Var(x|u)] + Var(E[x|u]) $$

求条件均值 $E[x|u]$
给定 $u$ 时，$u$ 是常数，$n_d$ 和 $n_r$ 是零均值高斯噪声（即对应部分的均值为0），因此：
$$ E[x|u] = E[g(\alpha u + n_d) + n_r \mid u] = g\alpha u $$
（因为 $E[n_d]=0$，$E[n_r]=0$）

求条件方差 $Var(x|u)$
同样给定 $u$，只有 $n_d$ 和 $n_r$ 是随机变量，且相互独立：
$$ 
\begin{split}
Var(x|u) 
&= Var(g\alpha u + g n_d + n_r \mid u)\\
&= g^2 Var(n_d) + Var(n_r)\\
&= g^2\sigma_d^2 + \sigma_r^2 
\end{split}
$$
($u$是常数，第一项的标准差为0)
注意，这个条件方差与 $u$ 无关，记为 $\sigma^2 = g^2\sigma_d^2 + \sigma_r^2$。

代回全方差公式
$$ \begin{aligned}
Var(x) &= E[Var(x|u)] + Var(E[x|u]) \\
&= E[\sigma^2] + Var(g\alpha u) \\
&= \sigma^2 + (g\alpha)^2 Var(u)
\end{aligned} $$

因为 $u \sim P(u^*)$，期望光电子数为 $u^*$，**泊松分布的方差等于均值**：
$$ Var(u) = E[u] = u^* $$

同时，期望的原始像素值（干净信号）为：
$$ x^* = E[x] = E[E[x|u]] = E[g\alpha u] = g\alpha u^* $$

因此 $u^* = \frac{x^*}{g\alpha}$。

代入可得：
$$ \begin{aligned}
Var(x) &= \sigma^2 + (g\alpha)^2 \cdot \frac{x^*}{g\alpha} \\
&= \sigma^2 + g\alpha \cdot x^*
\end{aligned} $$

令 $k = g\alpha$，最终得到：
$$ Var(x) = k x^* + \sigma^2 $$

---

### k-sigma变换推导

为了让网络输入具有一致的分布，PMRID提出了如下的**k-sigma变换公式**：
$$ f(x) = \frac{x}{k} + \frac{\sigma^2}{k^2} $$

将上述变换应用于原始像素值 $x$，可以得到变换后变量 $f(x)$ 的分布：
（泊松部分只与信号强度相关，因此变为x/k；高斯部分，原来的均值为0，但变换后+$\sigma^2/k^2$ 因此均值不为零，由于信号强度变为 $1/k$, 对应标准差变为$1/k^2$）
$$ f(x) \sim P\left(\frac{x^*}{k}\right) + \mathcal{N}\left(\frac{\sigma^2}{k^2}, \frac{\sigma^2}{k^2}\right) $$

**近似归一化**当光照条件不太差时，泊松分布 $P(\lambda)$ 可以用高斯分布 $\mathcal{N}(\lambda, \lambda)$ 来近似。代入上式得到：

$$ \begin{aligned}
f(x) &\sim P\left(\frac{x^*}{k}\right) + \mathcal{N}\left(\frac{\sigma^2}{k^2}, \frac{\sigma^2}{k^2}\right) \\
&\approx \mathcal{N}\left(\frac{x^*}{k}, \frac{x^*}{k}\right) + \mathcal{N}\left(\frac{\sigma^2}{k^2}, \frac{\sigma^2}{k^2}\right) \\
&= \mathcal{N}\left(\frac{x^*}{k} + \frac{\sigma^2}{k^2}, \frac{x^*}{k} + \frac{\sigma^2}{k^2}\right) \\
&= \mathcal{N}\left(f(x^*), f(x^*)\right)
\end{aligned} $$

经过k-sigma变换后，带噪数据 $f(x)$ 的分布 $\mathcal{N}(f(x^*), f(x^*))$ 在形式上已经**只依赖于干净的变换值 $f(x^*)$**，而不再直接依赖复杂的ISO增益参数 $g$。网络只需要学习这个固定分布的噪声，极大降低了对不同噪声水平的适应负担。
