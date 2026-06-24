---
type: artical
status: draft
tags:
  - machine-learning
  - backbone
  - transformer
rating: 0
create: 2026-06-22
update: 2026-06-23

publish: 2017-01-01
url: https://arxiv.org/pdf/1706.03762
---
Transformer 相关的介绍已经很多了，可以参考这个：[Transformer模型详解（图解最完整版） - 知乎](https://zhuanlan.zhihu.com/p/338817680)

值得注意的一点是：transformer 模型很大，训练成本高；同时用于训练的数据也非常多，因此一版预训练只跑一个epoch 就得到很好的效果：[[1906.06669] One Epoch Is All You Need](https://ar5iv.labs.arxiv.org/html/1906.06669)

图像领域也有类似的 1 Epoch 训练的研究（这里不深入了），例如EMP-SSL：[2304.03977](https://arxiv.org/pdf/2304.03977)

# Transformer 及其变体

为什么有的模型有编码器+解码器，有的模型只有编码器，有的只有解码器？

- **编码器（Encoder）**：双向注意力，即每个词都可以同时“看到”句子中所有的词（没有因果）
- **解码器（Decoder）**：它使用因果或单向注意力，即生成下一个词时，只能看到已经生成的词（左边），不能“偷看”未来的词

编码器跟解码器的特点天然适配不同类型的任务：
1. 理解：可以一次性看到全部信息，只使用编码器就够了（这里需求输出的结果比较简答。如果输出复杂结果，还需要解码器来生成）。例如：
	- 垃圾邮件分类
	- 识别、分割、检测等（[[ViT]]）
2. 生成：根据已有的输入续写，解码器比较合适。例如：
	- chatGPT
3. 转换：这是transformer 最初的目标，将一种语言翻译成另外一种语言，他需要先解码器理解，随后编码器生成。例如：
	- 翻译
	- 图像描述
