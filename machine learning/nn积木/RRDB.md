
```
Input (64 channels)
      │
      ├───────────────────────────────────────────┐
      │                                           │
      ▼                                           │
┌─────────────────────────────────────────────────┐
│  Dense Block                                    │
│  ┌─────────────────────────────────────────┐    │
│  │ Conv1 (3×3, 64→32) + LReLU              │    │
│  │      │                                   │    │
│  │      ▼                                   │    │
│  │ Concat(Input + Conv1) → 96 ch            │    │
│  │      │                                   │    │
│  │      ▼                                   │    │
│  │ Conv2 (3×3, 96→32) + LReLU              │    │
│  │      │                                   │    │
│  │      ▼                                   │    │
│  │ Concat(Input + Conv1 + Conv2) → 128 ch   │    │
│  │      │                                   │    │
│  │      ▼                                   │    │
│  │ Conv3 (3×3, 128→32) + LReLU             │    │
│  │      │                                   │    │
│  │      ▼                                   │    │
│  │ Concat(Input + Conv1 + Conv2 + Conv3)    │    │
│  │      │         → 160 ch                  │    │
│  │      ▼                                   │    │
│  │ Conv4 (3×3, 160→32) + LReLU             │    │
│  │      │                                   │    │
│  │      ▼                                   │    │
│  │ Concat(Input + Conv1+Conv2+Conv3+Conv4)  │    │
│  │      │         → 192 ch                  │    │
│  │      ▼                                   │    │
│  │ Conv5 (3×3, 192→64) + LReLU             │    │
│  └─────────────────────────────────────────┘    │
│                                                  │
│           │                                      │
│           ▼                                      │
│      × β (残差缩放系数 = 0.2)                    │
│           │                                      │
└───────────┼──────────────────────────────────────┘
            │
            ▼
      ┌───────────┐
      │   相加    │ ◄── 与 Input 做元素级加法
      └───────────┘
            │
            ▼
      Output (64 channels)
```


### 逐层通道数变化表

|步骤|操作|输入通道|输出通道|说明|
|---|---|---|---|---|
|0|Input|—|64|RRDB 输入|
|1|Conv1 + LReLU|64|32||
|2|Concat(Input, Conv1)|64+32=96|—|密集连接开始|
|3|Conv2 + LReLU|96|32||
|4|Concat(Input, Conv1, Conv2)|96+32=128|—||
|5|Conv3 + LReLU|128|32||
|6|Concat(Input, Conv1, Conv2, Conv3)|128+32=160|—||
|7|Conv4 + LReLU|160|32||
|8|Concat(Input, Conv1, Conv2, Conv3, Conv4)|160+32=192|—||
|9|Conv5 + LReLU|192|64|恢复原通道数|
|10|× β|64|64|残差缩放 (β=0.2)|
|11|Add with Input|64|64|局部残差连接|

### 📌 三个关键设计要点

**1. 密集连接（Dense Connection）**

- 每个卷积层的输入都包含了**之前所有层的输出**
    
- 好处：最大化信息流动，缓解梯度消失，让 23 个 RRDB 堆起来也不会丢信息
    

**2. 残差缩放（Residual Scaling，β=0.2）**

- Dense Block 的输出乘以 **0.2** 后再与输入相加
    
- 好处：防止深层网络的梯度爆炸，让训练极其稳定。这是 ESRGAN 能堆 23 个 RRDB 的核心技巧
- 正常的残差连接有BN 归一化，因此可以堆很深也不会出现梯度问题。这里去掉了BN(防止改变数据分布，画面变灰等)，因此会出现梯度问题，试验了也残差衰减到0.2的技巧。
- 值得一提的是，即使是降噪任务，新模型也倾向于用**LN（Layer Normalization）** 或**IN（Instance Normalization）** 替代BN。例如2025年的CP-UNet就用LN替代了BN来防止梯度问题[](https://ui.adsabs.harvard.edu/abs/2025arXiv250213395H/abstract)。因为BN对Batch Size非常敏感，而Raw图分辨率高导致Batch Size通常很小，BN的统计量不准反而会造成训练不稳定。
    

**3. 局部残差 + 全局残差（Residual-in-Residual）**

- **局部残差**：每个 RRDB 内部的 Dense Block 输出与输入相加
    
- **全局残差**：整个 RRDB 堆叠的输入（`conv_first`）与输出（`conv_body`）相加
    
- 这就是名字 **Residual-in-Residual** 的由来——残差里面套残差