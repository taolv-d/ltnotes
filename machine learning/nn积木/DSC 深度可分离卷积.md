depthwise separable convolution 深度可分离卷积

可以理解成：把一次普通卷积，拆成两步做:先做“每个通道内部的空间提取”，再做“跨通道融合”

1. depthwise conv  
    每个输入通道各自用一个卷积核做空间卷积，不和别的通道混合。
    
2. pointwise conv  
    再用 1 x 1 卷积，把这些通道线性组合起来，完成通道间的信息融合。

所以它比普通卷积便宜很多，算力和参数都更小，非常适合移动端模型。

# 与普通卷积对比
**普通卷积**  
假设输入特征图是：

- 高宽：H x W
- 输入通道数：C_in

如果要输出 C_out 个通道，卷积核不是一个，而是 **C_out 组卷积核**。

每一个卷积核的形状是：

- K x K x C_in

所以整层卷积的权重张量形状是：

- K x K x C_in x C_out

---

**Depthwise Separable Convolution**  
它拆成两步。

**第 1 步：Depthwise Conv**  
如果输入是 H x W x C_in，那么：

- 不再用 K x K x C_in 的大卷积核
- 而是**每个输入通道单独用一个 K x K x 1 卷积核**

所以一共有：

- C_in 个卷积核
- 每个大小 K x K x 1

权重形状可理解为：

- K x K x C_in

输出通道数还是：

- C_in

因为每个输入通道各卷各的，不做通道混合。

**第 2 步：Pointwise Conv**  
然后接一个 1x1 卷积来混合通道。

如果输入是 H x W x C_in，输出想要 C_out：

- 每个卷积核大小：1 x 1 x C_in
- 一共 C_out 个卷积核

所以参数量：

- 1* 1 * C_in * C_out


---

例子：

输入：

- 112 x 112 x 3

目标输出：

- 112 x 112 x 64

如果用普通卷积：

- 参数量：3* 3* 3* 64 = 1728

如果用 depthwise separable conv：

- depthwise：3* 3* 3 = 27
- pointwise：1* 1* 3* 64 = 192
- 总计：219