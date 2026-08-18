---
type: artical
status: todo
tags:
  - 
rating: 0
create: 2026-07-24
update: 2026-08-18
publish: 2022-01-01
url: https://ar5iv.labs.arxiv.org/html/2203.11483
---
# 解决问题
- **精细结构恢复困难**：网、绳子等困难场景->级联循环网络，从粗到精逐步细化
- **非理想校正带来的错配**：内外参飘移、异构双目->可变形窗口
- **困难场景泛化能力不足**：无纹理、重复纹理、遮挡等->数据集扩充，提升泛化能力
# 网络架构

## 级联循环架构

![[attachments/Pasted image 20260724222826.png]]

### 自适应分组相关层（AGCL）

![[attachments/Pasted image 20260724222836.png]]

# 网络训练


https://jishuzhan.net/article/2012030050964127746#%EF%BC%88%E5%9B%9B%EF%BC%89%E6%96%B9%E6%B3%95%E2%80%94%E2%80%94%E8%AE%BA%E6%96%87%E4%B8%AD%E7%9A%84%E6%A0%B8%E5%BF%83%EF%BC%81