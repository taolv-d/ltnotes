---
type: note
status: done
tags:
  - camera
  - sensor
  - adc
rating: 0
create: 2026-05-17
update: 2026-06-23
---
# 四种 ADC 对比

| Type           | 原理图                                              | 备注                                                                                                                      |
| -------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| $\Delta\Sigma$ | ![[attachments/Pasted image 20260516095853.png]] | ![[attachments/Pasted image 20260516095943.png]]<br>上图描述的原理是：长度 $L$ 尺子5个，高度 $x$ 的书8个，此时每本书的高度就是 $5*L/8$ 。这里尺子就是DAC 参考输出 |
| 连续近似           | ![[attachments/Pasted image 20260516100302.png]] | 由高位到低位依次比较，类似二分法<br>（需要N次比较，确定每一位是还是1）<br><br>左侧比较电路需要剪掉DAC的输出                                                          |
| single-slope   | ![[attachments/Pasted image 20260516100226.png]] | 由0~$2^N$ 一个值一个值的比（使用计数器）。值越大时间比较时间越长。<br><br>左侧比较电路只输出一个停止信号就行                                                          |
| Flash          | ![[attachments/Pasted image 20260516100953.png]] | 类似数组，只需要比较1次                                                                                                            |

# 那种 ADC 好

![[attachments/Pasted image 20260516101043.png|427]]

analog 的一篇技术报告比较了各种ADC：[A Simple ADC Comparison Matrix | Analog Devices](https://www.analog.com/en/resources/technical-articles/a-simple-adc-comparison-matrix.html)比较的表格也摘抄到这里：

|                                         | FLASH (Parallel)                                                                | SAR                                                                               | DUAL SLOPE (Integrating ADC)                                                                   | PIPELINE                                                                                   | SIGMA DELTA                                                                                                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Pick This Architecture if you want:** | Ultra-High Speed when power consumption not primary concern?                    | Medium to high resolution (8 to 20 bits), 5Msps and under, low power, small size. | Monitoring DC signals, high resolution, low power consumption, good noise performance ICL7106. | High speeds, few Msps to 100+ Msps, 8 bits to 16 bits, lower power consumption than flash. | High resolution, low to medium speed, no precision external components, simultaneous 50Hz/60Hz rejection, digital filter reduces anti-aliasing requirements. |
| **Conversion Method**                   | N bits - 2N - 1 Comparators Caps increase by a factor of 2 for each bit.        | Binary search algorithm, internal circuitry runs higher speed.                    | Unknown input voltage is integrated and value compared against known reference value.          | Small parallel structure, each stage works on one to a few bits.                           | Oversampling ADC, 5Hz to 60Hz rejection programmable data output.                                                                                            |
| **Encoding Method**                     | Thermometer Code Encoding                                                       | Successive Approximation                                                          | Analog Integration                                                                             | Digital Correction Logic                                                                   | Over-Sampling Modulator, Digital Decimation Filter                                                                                                           |
| **Disadvantages**                       | Sparkle codes/metastability, high power consumption, large size, expensive.     | Speed limited to ~5Msps. May require anti-aliasing filter.                        | Slow Conversion rate. High precision external components required to achieve accuracy.         | Parallelism increases throughput at the expense of power and latency.                      | Higher order (4th order or higher) - multibit ADC and multibit feedback DAC.                                                                                 |
| **Conversion Time**                     | Conversion Time does not change with increased resolution.                      | Increases linearly with increased resolution.                                     | Conversion time doubles with every bit increase in resolution.                                 | Increases linearly with increased resolution.                                              | Tradeoff between data output rate and noise free resolution.                                                                                                 |
| **Resolution**                          | Component matching typically limits resolution to 8 bits.                       | Component matching requirements double with every bit increase in resolution.     | Component matching does not increase with increase in resolution.                              | Component matching requirements double with every bit increase in resolution.              | Component matching requirements double with every bit increase in resolution.                                                                                |
| **Size**                                | 2N - 1 comparators, Die size and power increases exponentially with resolution. | Die increases linearly with increase in resolution.                               | Core die size will not materially change with increase in resolution.                          | Die increases linearly with increase in resolution.                                        | Core die size will not materially change with increase in resolution.                                                                                        |

# ADC 应该放在哪？

![[attachments/Pasted image 20260516101141.png|404]]

# ADC 位数选择

**噪声对 ADC 的影响**：参考[[芝加哥大学：Noise DR Bit Depth in Digital SLRs]]
> 这里主要考虑噪声，在全黑环境下，sensor只有暗噪声，此时可以认为是最弱的噪声水平。
> 1. 如果ADC的分辨率比这个噪声水平（图像标准差）精细很多，更精细的分辨也是噪声，对提升图像质量没有帮助。
> 2. 如果ADC的分辨率比这个噪声水平粗略很多，此时ADC分辨率不足，就会出现等高线效应。
> 3. 因此最合适的ADC位数应该是比噪声水平稍精细一点。

**性能、功耗与成本的平衡**：参考这篇博客：[告别‘黑盒’：深入CMOS Sensor内部，看懂ADC位数、模拟增益与HDR背后的硬件逻辑-CSDN博客](https://blog.csdn.net/weixin_29237635/article/details/159608080)
>1. CMOS的制程（>=40 nm）时ADC的信噪比限制在70dB左右。70dB 对应十进制为3162，对应用12bit ADC（4096）刚好覆盖。即便更高精度ADC不会有提升
>2. 面积成本，增加ADC位数会增加面积，进一步影响良率，从而抬高成本
>	1. ADC阵列占Sensor核心面积的15%-25%。
>	2. 每增加1bit精度ADC单元面积需扩大2.5倍（保持相同采样率）。
>3. 功耗：
>	1. 功耗可以用如下模型近似：$0.5*2^{N_bit}*fps*1e^{-6}$ mW。对应12bit 30fps 约61.4mW ；12bit 30fps 约245.8mW
>	2. 手机这种功耗敏感的设备，Sensor宁可采用多次曝光HDR方案也不盲目追求高bit ADC。
