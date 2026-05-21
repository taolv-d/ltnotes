参考资料：

中文翻译：https://ltfa1l.top/2023/09/23/system/Heap/understanding\_glibc\_malloc/

英文原文：[Understanding glibc malloc](https://sploitfun.wordpress.com/2015/02/10/understanding-glibc-malloc/comment-page-1/)

**重点：**

Linux 下  glibc 2.23 版本 的 ptmalloc2 分配器（即 `malloc/free` 的实现）

* 内存分配的设计哲学（效率 vs. 安全性）

* 关键数据结构（如 `chunk`、`bin`、`arena`）

* 分配/释放的算法流程

* 多线程优化（`arena` 竞争管理）

* 安全漏洞的潜在来源（如堆溢出、UAF）

***

### 二、关键内容详解

#### 1. 核心数据结构

* Chunk（内存块）：`malloc` 管理的基本单位，包含元数据（`prev_size` 和 `size`、标志位等）和用户数据区。

  * Chunk的大小不是随意的，通常根据用户请求的大小，+ 元数据 + 满足内存对齐需求来确定

  * 程序起来时，就已经有一个chunk 位于主arena，此时top chunk记录了内存最顶端的位置。后续malloc则从top chunk分割。此时所有bins都是空的

  * 如果第一次请求内存超过 top chunk大小，但是不足以出发 mmap，此时使用brk系统调用分配

  * 否则使用mmap加载（需要虚拟内存到物理内存的映射）

  * 对于非主线程：

    1. 线程第一次调用 `malloc` 时，会创建线程本地 arena

    2. 初始时通过 `mmap` 分配一块内存（通常 64MB 或 1MB，取决于配置）

    3. 设置线程的 `top chunk`，流程与主线程类似但使用 `mmap` 而非 `brk`

  * Chunk 中记录了自己的size，理论上可以从顶端一直遍历到低端所有chunk

  * Chunk 中记录了上一个chunk的size即prev\_size，理论上可以反向遍历。但实际不行，原因是只有上一个chuak为free时才成立，即全都free才能反向遍历

* Bins（回收站）：用于缓存空闲 chunk 的链表，分为：

  * Fast bins：单链表，LIFO，小内存（默认 ≤ 64B）快速分配，不合并相邻空闲块。

  * Small bins：双向循环链表，FIFO，管理较小尺寸（< 512B）的 chunk。

  * Large bins：管理大 chunk，按尺寸范围分组，支持最佳匹配搜索。

  * Unsorted bin：临时存放释放的 chunk，作为分配时的“缓冲区”。

  * 当请求大小跟记录的chunk大小不匹配时：

    * Fast small 通常不分割

    * Large unsorted 分割

    * 分割提高了内存利用率，避免碎片，但也降低了性能

* Arena（分配区）：每个线程的独立内存管理区域，减少锁竞争。

  * Main arena：主线程使用，通过 `brk` 扩展堆。

  * Thread arena：子线程通过 `mmap` 创建非连续堆。

  * Arena 主要解决多线程malloc竞争的问题，如果没有 arena，所有线程的malloc都要等同一把全局锁。有了arena大大避免了竞争。

  * Arena 数不是每个线程有一个，一般是1+处理器核心\*2（与处理器核心有关，原因是多核并行才会触发malloc抢占，单核并行同一时刻只能有一个程序malloc，不用担心抢占）。但是线程数很多时，会出现多个线程共用一个arena的情况，此时在多核CPU上仍会抢占，但受影响的线程数量较少。

  * Arena 问题：

    * 内存碎片，每个arena剩余的内存加起来很多，但分散，此时请求相当的内存会失败

    * 伪共享：考虑两个不同的线程共用一个arena，这两个线程在两个不同的CPU上运行（cache不共享），造成arena需要不断在两个cache之间同步

#### 2. 分配流程（malloc）

1. 检查 fast bins → 若匹配则直接分配。

2. 检查 small bins → 若匹配则分配。

3. 遍历 unsorted bin，同时将不符合的 chunk 归类到 small/large bins。

4. 搜索 large bins → 寻找最小匹配 chunk。

5. 若仍失败，则通过 `top chunk` 切割或调用 `syscall`（`brk`/`mmap`）扩展堆。

#### 3. 释放流程（free）

1. 检查是否与 fast bins 匹配 → 是则插入（不合并）。

2. 否则合并相邻空闲 chunk，放入 unsorted bin。

3. 若释放块过大（>64KB），可能通过 `madvise` 返还部分内存给内核。

4. Consolidation：定期合并 fast bins 中的 chunk 以避免碎片。

#### 4. 多线程优化

* Per-thread arena：线程首次分配时创建自己的 arena（最多 `#cores * 2` 个）。

* 锁竞争：通过原子操作和 arena 分区减少全局锁争用。

***

**完整分配过程**

```plain&#x20;text
用户空间应用
    │ malloc/free
    ▼
glibc malloc
    │ 管理bins、arenas、tcache
    │ 决定使用brk还是mmap
    ▼
系统调用层
    │ brk() / mmap() / munmap()
    ▼
内核虚拟内存管理
    │ 创建/销毁VMA，更新页表
    ▼
缺页异常处理
    │ 按需分配物理页面
    ▼
内核物理内存管理
    │ 伙伴系统分配/释放页面
    │ 页面回收、交换
    ▼
硬件MMU
    │ TLB、页表遍历
    ▼
物理内存
```

**什么时候调用brk, 什么时候mmap**

1. `brk` 主要在主arena的小块内存申请（< mmap\_threshold）且top chunk不足时调用。

2. mmap在三种情况下调用：

   1. thread arena的所有堆扩展

   2. 任何arena的大内存分配（≥ mmap\_threshold）

   3. 特殊情况下主arena无法通过brk满足请求时

```plain&#x20;text
malloc(size):
    ↓
1. 如果 size <= tcache_max_size:
     从 tcache 分配（无系统调用）
     
    ↓
2. 如果 size <= fastbin_max_size:
     从 fastbin 分配（无系统调用）
     
    ↓  
3. 如果 size < mmap_threshold（默认128KB）:
     ↓
     3.1 从已有的bins中查找（无系统调用）
          ↓
     3.2 如果找不到且是主arena:
          从top chunk切割（可能触发brk）
          ↓
     3.3 如果找不到且是thread arena:
          通过mmap扩展heap
          
    ↓
4. 如果 size >= mmap_threshold:
     直接使用mmap分配（总是调用mmap）
```

### 安全视角的延伸

文章为堆漏洞利用打下基础，解释了：

* 堆溢出：覆盖相邻 chunk 的元数据（如 size）可导致任意地址读写。

* Use-After-Free（UAF）：释放后未清零指针，再分配可能篡改数据。

* Fast bin attack：通过篡改 fast bin 链表实现任意地址分配。

* Unlink 漏洞：早期 glibc 在合并 chunk 时未充分验证双向链表完整性，可导致写内存原语（现代 glibc 已加强验证）。

