---
type: note
status: done
tags:
  - computer
  - architecture
rating: 0
create: 2026-05-21
update:
---
1. SRAM 静态存储器，使用锁存器存储，通常使用6个晶体管保存一个 bit

2. DRAM 动态存储，使用电容存储一个bit，由于电容会漏电，因此要不断刷新，同时读取后要马上再写回去

3. GDDR，Graphics Data RAMs 是提供数据带宽和时钟频率满足GPU需求

4. Flash&#x20;

   闪存是一种EEPROM，其与DRAM最大的不同在于：

   1. 读取闪存是顺序的，一次读取一整页内容，其读取速度在DRAM和硬盘之间。

   2. 闪存必须先被擦除才能重新写入，擦除以块为单位，其写入速度在DRAM和硬盘之间，但比读取速度优势小得多。

   3. 闪存是非易失性的，待机功耗很小。

   4. 闪存的块只能被写入有限次，所以需要平衡写入负载。

   5. 闪存的价格在SDRAM和硬盘之间。

   DRAM和闪存芯片都有冗余的块用于替换损坏的块。
