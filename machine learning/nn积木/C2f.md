
C2f = 先分流，再逐步加工，再把多路特征拼起来融合

```
class C2f(nn.Module):
	def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
		self.c = int(c2 * e)
		self.cv1 = Conv(c1, 2 * self.c, 1, 1)
		self.cv2 = Conv((2 + n) * self.c, c2, 1)
		self.m = nn.ModuleList(
			Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
			for _ in range(n)
		) 

	def forward(self, x):
		y = list(self.cv1(x).chunk(2, 1))
		y.extend(m(y[-1]) for m in self.m)
		return self.cv2(torch.cat(y, 1))
```

**先说它到底在干嘛**  
假设输入是一个特征图 x，C2f 会做 4 件事：

1. 用一个 1x1 Conv 先调整通道数
2. 把结果沿着通道维切成两份
3. 其中一份接着过多个 Bottleneck，每次都生成新的特征
4. 把“前面保留下来的特征 + 后面新生成的特征”全部拼接起来，再用一个 1x1 Conv 融合输出

```
输入 x
  |
  v
cv1: 1x1 Conv
  |
  v
按通道切成两份
  |--------------------> y0
  |
  |--------------------> y1
                           |
                           v
                    Bottleneck 1
                           |
                           v
                          y2
                           |
                           v
                    Bottleneck 2
                           |
                           v
                          y3
                           |
                           v
                    Bottleneck 3
                           |
                           v
                          y4

最后把这些特征全部拼接:
[y0, y1, y2, y3, y4]
  |
  v
cv2: 1x1 Conv 融合
  |
  v
输出 out

```

所以它不是一条直线，而更像一个“小型特征收集器”。

**按张量流动来理解**  

`y = list(self.cv1(x).chunk(2, 1))`

- 先 cv1(x)，得到 2*self.c 个通道
- 然后按通道切成两半
- chunk(2, 1) 里的 1 表示沿 channel 维切

所以现在 y 里面先有两个分支：

- y[0]
- y[1]

然后这句：
`y.extend(m(y[-1]) for m in self.m)`

- 取当前最后一个分支
- 送进一个 Bottleneck
- 得到新特征
- 再继续送进下一个 Bottleneck

如果 n=3，那 y 最后会有：

- 最初切出来的两份
- 再加上 3 个新生成的特征

一共 2 + n = 5 份特征。

最后：
`return self.cv2(torch.cat(y, 1))`

就是把这 5 份特征在通道维拼起来，再做一次 1x1 Conv 融合成输出。

**为什么这样设计**  
核心目的有 3 个：

- 保留原始信息
- 逐步提炼新信息
- 让不同层次的特征一起参与输出

普通“直筒子”结构常常是：
`x -> block1 -> block2 -> block3 -> out`
这样最后输出主要依赖最末端特征。

而 C2f 更像：
`原始一点的特征 中间一点的特征 更深一点的特征 都被保留下来一起用`
所以它有点像“边加工边收集”。


C2f 比较擅长做特征复用，速度和效果平衡得好

**里面的 Bottleneck 是什么**  
C2f 里面不是直接堆卷积，而是堆了多个 Bottleneck。[[Bottleneck]]
- 外面一层“大分流 + 汇总”
- 里面串了一些“小残差块”


**你可以先记住这个最实用的理解**  
以后你在 YOLOv8 里看到 C2f，脑子里直接翻译成：

一个高效的多分支特征复用模块
边保留旧特征，边生成新特征，最后一起融合
