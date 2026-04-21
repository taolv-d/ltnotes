SE 全称是 **Squeeze-and-Excitation**。

假设输入是：`H x W x C`

SE 做这几步：

```
1. Global Average Pooling
   H x W x C → 1 x 1 x C

2. FC / 1x1 conv 降维
   C → C/r

3. 激活

4. FC / 1x1 conv 升维
   C/r → C

5. sigmoid / hard sigmoid
   得到每个通道的权重

6. 原特征逐通道乘权重

```
直观理解：

- 网络自己判断哪些通道重要
- 重要通道放大
- 不重要通道压低
