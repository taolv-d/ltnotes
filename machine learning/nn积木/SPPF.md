SPPF 是 YOLO 里一个很经典的“**扩大感受野、融合多尺度上下文**”的小模块。
SPPF = 用多次池化，让网络在不改变特征图大小的前提下，看见更大范围的信息

**它解决什么问题**  
卷积天然更擅长看局部。层数加深以后虽然感受野会变大，但网络有时还是希望在某个阶段显式地“看看更大范围”。

比如做检测时，一个点附近的信息有时不够，需要知道：

- 物体整体大概多大
- 周围上下文是什么
- 大范围结构长什么样

**画一个简单数据流图**  
```
x [B,1024,20,20]
  |
  v
cv1: 1x1 Conv
  |
  v
y0 [B,512,20,20]
  |
  +--> MaxPool(5*5) -> y1 [B,512,20,20]
          |
          +--> MaxPool(5*5) -> y2 [B,512,20,20]
                  |
                  +--> MaxPool(5*5) -> y3 [B,512,20,20]

把 y0, y1, y2, y3 拼接
=> [B, 2048, 20, 20]
  |
  v
cv2: 1x1 Conv 融合
  |
  v
out [B,1024,20,20]
```


```
class SPPF(nn.Module):
	def __init__(self, c1, c2, k=5, n=3, shortcut=False):
		c_ = c1 // 2
		self.cv1 = Conv(c1, c_, 1, 1, act=False)
		self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
		self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

	def forward(self, x):
		y = [self.cv1(x)]
		y.extend(self.m(y[-1]) for _ in range(3))
		y = self.cv2(torch.cat(y, 1))
		return y
```


你可以把它拆成 3 步。

**第 1 步：先压缩通道**

`self.cv1 = Conv(c1, c_, 1, 1, act=False)`

- 用一个 1x1 Conv 先把**通道**数减半
- 降低后续计算量

**第 2 步：连续做最大池化**

`self.m = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)`

- kernel_size=5
- stride=1
- padding=2

所以池化后 **高宽不会变**。

然后在前向里：

`y = [self.cv1(x)] y.extend(self.m(y[-1]) for _ in range(3))`

- 先保留原始特征 y0
- 对 y0 做一次池化得到 y1
- 对 y1 再池化得到 y2
- 对 y2 再池化得到 y3

所以最后有 4 份特征：

- 原始特征
- 一次池化后的特征
- 两次池化后的特征
- 三次池化后的特征

这些特征都还是同样的高宽。

**第 3 步：拼接并融合**

`y = self.cv2(torch.cat(y, 1))`

- 把 4 份特征在通道维拼起来
- 再用 1x1 Conv 融合

所以最终输出又变回目标通道数 c2。

**它为什么能扩大感受野**  
虽然每次池化都不改变尺寸，但池化核是 5x5，它会看邻域。
- 多次池化后的特征更偏大范围上下文
- 把它们拼起来，就兼顾局部和全局一点的感知

**它和 SPP 的关系**  
经典 SPP 往往是并联多个不同核的池化，比如：

- 5x5
- 9x9
- 13x13

然后拼接。

而 SPPF 是一种更快的近似写法：

- 不用并联很多大核
- 改成连续几次相同的小核池化
- 达到类似的多尺度效果

所以它名字里有个 F，就是 Fast。