---
type: note
status: done
tags:
  - machine-learning
  - nn-block
rating: 0
create: 2026-04-14
update: 2026-08-27
---
Neck 是**目标检测里很常见的一段中间结构**：把Backbone（骨干网络）在不同层级提取到的、分辨率和语义信息各异的特征图，有效地融合在一起，让最终用于预测的特征图既能“看得广”（包含全局上下文），也能“看得细”（保留局部细节），从而让YOLO能同时擅长检测大、中、小各种目标。
![[attachments/Pasted image 20260827212151.png]]

