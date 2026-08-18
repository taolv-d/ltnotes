---
type: artical
status: done
tags:
  - camera
  - isp
  - mllm
rating: 0
create: 2026-05-06
update: 2026-08-18
publish: 2025
url: https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Multimodal_Large_Language_Model-Guided_ISP_Hyperparameter_Optimization_with_Dynamic_Preference_ICCV_2025_paper.pdf
---
原文：[Multimodal Large Language Model-Guided ISP Hyperparameter Optimization with Dynamic Preference Learning](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Multimodal_Large_Language_Model-Guided_ISP_Hyperparameter_Optimization_with_Dynamic_Preference_ICCV_2025_paper.pdf)
本文的工作聚焦于如何让大模型来做IQ调试，工程上很有意义。不过当今AI发展迅猛，本文面对的很多问题可能会随着模型能力的提升而消失。不过解决问题的思路值得学习。
![[attachments/Pasted image 20260506100800.png]]
本文实验时模型的能力应该还比较一般（GPT-4V、LLaVA-7B等），因此整体架构还比较复杂。
**上图是推理时的架构，主要分为几步：**
1. 多模态大模型（MLLM）接收图像跟任务要求输出描述（描述内容类似：图像偏暗、噪声太大等）
2. 图像、MLLM描述送入Clip 编码，形成编码信息
3. 编码信息送入Agent A, Agent B，生成两组ISP参数，通过一个决策函数 $f$ 来选择一个（传统指标，如噪声强度等）
	- 这里两个Agents 是用不同权重的模型（MLP）
	- 决策函数 $f$ 应该是用的客观指标
4. 主Agent对生成的ISP参数微调
5. 应用到ISP
**训练：**
训练主要针对 Agent A，Agent B 进行，采用强化学习（ISP 参数通常不可微，强化学习生成奖励信号，无需梯度信息）。
- MLLM、Clip等应该不参与训练
- Clip 编码输出送到一个Agent网络，随后用强化学习算法训练该网络
**微调：**
- 微调主要是针对主Agent，针对特定IQ任务微调。（前面的训练可能会偏离某些IQ任务）

