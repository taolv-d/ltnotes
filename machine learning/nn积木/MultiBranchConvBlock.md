MultiBranchConvBlock 是一个**轻量的多分支卷积块**，作用是用比较低的成本同时看不同感受野。

它的结构非常直接：输入 x 之后，并行走 3 个分支，每个分支都是 **depthwise conv + 激活函数**。

3 个分支分别是：

- 3x3 depthwise conv, dilation=1
- 3x3 depthwise conv, dilation=2
- 5x5 depthwise conv, dilation=1

每个分支前都用了 ReflectionPad2d 来补边。  
3 个分支输出之后，**直接相加**：

out = branch1(x) + branch2(x) + branch3(x)

然后再过一个：

- 1x1 conv

做通道融合，得到最终输出。

这里几个关键点是：

**1. 为什么是 depthwise conv**  
它不是普通卷积，而是 groups=channels 的 depthwise conv。  
意思是每个通道单独卷，不做通道间混合，所以计算量比较低。

**2. 为什么要多分支**  
因为不同分支看到的空间范围不一样：

- 3x3 d=1 看局部细节
- 3x3 d=2 感受野更大
- 5x5 d=1 也能覆盖更大邻域，但采样方式不同

把它们加起来，相当于同时提取多尺度局部模式。

**3. 为什么最后接 1x1 conv**  
前面的 depthwise conv 只是在每个通道里做空间处理，通道之间没怎么“交流”。  
最后的 1x1 conv 负责把这些分支结果重新融合。
