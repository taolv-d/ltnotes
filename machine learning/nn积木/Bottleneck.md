
Bottleneck = 用较少的计算做一次“特征加工”，并且常常带一条捷径连接（shortcut）

```
class Bottleneck(nn.Module): 
	def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
		c_ = int(c2 * e)
		self.cv1 = Conv(c1, c_, k[0], 1)
		self.cv2 = Conv(c_, c2, k[1], 1, g=g)
		self.add = shortcut and c1 == c2

	def forward(self, x):
		return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
```

**它到底在做什么**  
它的主路很简单：

1. cv1
2. cv2

也就是输入 x 先经过两次卷积变换，得到一个新特征。

如果满足条件，还会走一条捷径：

`输出 = x + 变换后的x`

这条“直接把输入加回来”的路，就叫：

- shortcut
- 或 skip connection
- 或 residual connection

所以你可以把 Bottleneck 看成：

两层卷积 + 可选残差连接

**为什么叫 Bottleneck**  
这个词原意是“瓶颈”。  
在神经网络里，它通常表示中间会先把通道压一压，再恢复或变换出去。

看这句：

`c_ = int(c2 * e)`

它定义了一个中间隐藏通道数 c_。

比如：

- 输入输出通道想要是 256
- e=0.5
- 那么中间通道 c_ = 128

于是流程就变成：

- 先把特征变到更小的中间空间
- 在这个空间里做计算
- 再映射到输出通道

这就是“瓶颈”的感觉：中间收窄一下。

这样做的目的通常是：

- 减少参数量
- 减少计算量
- 仍然保留足够的表达能力

**我们逐行拆一下**

**1. 第一层卷积 cv1**

`self.cv1 = Conv(c1, c_, k[0], 1)`

作用：

- 把输入通道从 c1 变成中间通道 c_
- 做一次特征提取

如果 c1=128, c2=128, e=0.5，那这里就是：

- 128 -> 64

**2. 第二层卷积 cv2**

`self.cv2 = Conv(c_, c2, k[1], 1, g=g)`

作用：

- 把中间通道再变成输出通道 c2
- 再做一次特征变换

继续上面的例子：

- 64 -> 128

所以整体像：

`x -> Conv(128 -> 64) -> Conv(64 -> 128) -> out`

**3. 残差连接 add**

`self.add = shortcut and c1 == c2`

意思是只有在这两个条件同时满足时，才做残差：

- shortcut=True
- 输入通道数 c1 和输出通道数 c2 相等（相加操作的需求）


所以前向传播是：

`return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))`

如果能加，就走残差版本；不能加，就只输出主路结果。


所以它有两个典型特征：

- 中间通道更窄
- 有残差连接

Bottleneck 里的 cv1 和 cv2 也**不是“只有纯卷积”**，实际包含：

- nn.Conv2d
- nn.BatchNorm2d
- activation，默认是 SiLU
