---
type: note
status: done
tags:
  - machine-learning
  - nn-block
rating: 0
create: 2026-04-14
update: 2026-08-26
---
C2f (CSP Bottleneck with 2 Convolutions，其中CSP是Cross Stage Partial)。是yolo中非常关键的一个模块。
C2f：先分流，再逐步加工，再把多路特征拼起来融合。保留旧特征，同时提取新特征。
“直筒子”结构`x -> block1 -> block2 -> block3 -> out`，主要依赖末端输出的特征。

C2f 内堆了多个 [[Bottleneck]]
![[attachments/Pasted image 20260826230538.png|370]]
