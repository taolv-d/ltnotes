depthwise separable convolution深度可分离卷积 可以理解成：把一次普通卷积，拆成两步做:先做“每个通道内部的空间提取”，再做“跨通道融合”

1. depthwise conv  
    每个输入通道各自用一个卷积核做空间卷积，不和别的通道混合。
    
2. pointwise conv  
    再用 1x1 卷积，把这些通道线性组合起来，完成通道间的信息融合。
    

在这个仓库里就是这样写的，models/net_torch.py (line 16)：

```
modules['depthwise'] = nn.Conv2d(
	in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False,
)
modules['pointwise'] = nn.Conv2d(
	in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True,
)
```

这里 groups=in_channels 就表示“每个通道自己卷自己”，这就是 depthwise；后面的 1x1 卷积就是 pointwise。

拿一个例子看更直观。假设输入是 32 个通道，输出是 64 个通道，卷积核大小 3x3：

- 普通卷积参数量：32 x 64 x 3 x 3 = 18432
- depthwise separable：
    - depthwise：32 x 3 x 3 = 288
    - pointwise：32 x 64 = 2048
    - 总共：2336

所以它比普通卷积便宜很多，算力和参数都更小，非常适合移动端模型。