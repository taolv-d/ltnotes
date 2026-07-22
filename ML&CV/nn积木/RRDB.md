---
type: note
status: done
tags:
  - machine-learning
  - nn-block
rating: 0
create: 2026-04-17
update: 2026-07-21
---

参考这篇笔记 [](../image%20super-resolution/2018%20RRDBNet%20ESRGAN%20Tencent.md)

RRDB使用减少，主要源于以下挑战和竞争：
- 性能并非最优：一些最新对比研究表明，在某些任务（如3D MRI超分）中，RRDB的性能落后于U-Net等模型。
- 计算开销大：其复杂的嵌套残差和密集连接结构带来了较高的计算成本。
- 被更新的架构替代：Transformer及各类轻量化网络（如SwinIR、Restormer等）在性能或效率上展现出更大优势，成为研究新宠。