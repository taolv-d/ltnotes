---
type: note
status: done
tags:
  - camera
  - sensor
  - adc
rating: 0
create: 2026-05-17
update:
---

| Type           | 原理图                                              | 备注                                                             |
| -------------- | ------------------------------------------------ | -------------------------------------------------------------- |
| $\Delta\Sigma$ | ![[attachments/Pasted image 20260516095853.png]] | ![[attachments/Pasted image 20260516095943.png]]               |
| single-slope   | ![[attachments/Pasted image 20260516100226.png]] | 由0~$2^N$ 一个值一个值的比（使用计数器）。值越大时间比较时间越长。<br><br>左侧比较电路只输出一个停止信号就行 |
| 连续近似           | ![[attachments/Pasted image 20260516100302.png]] | 由高位到低位依次比较，类似二分法<br>（需要N次比较，确定每一位是还是1）<br><br>左侧比较电路需要剪掉DAC的输出 |
| Flash          | ![[attachments/Pasted image 20260516100953.png]] | 类似数组，只需要比较1次                                                   |

**那种ADC 好**
![[attachments/Pasted image 20260516101043.png]]

**ADC 应该放在哪？**
![[attachments/Pasted image 20260516101141.png]]

**ADC 位数选择**