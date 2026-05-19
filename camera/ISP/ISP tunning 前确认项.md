
**camera & ISP**
- 产品应用场景及画质需求
- ISP调试结果交付方式
- ISP框图（先后顺序等，统计信息来源）
- Linear or HDR
- bit depth，W H, fps, Color pattern , raw format
- 是否需要使用OTP参数
- camera具备内置ISP?
- HDR sensor 曝光比及曝光时间的限定
- WDR sensor 解压节点
- gain range、exposure time range
- 模组个体硬件性能一致性评估
- IR
- 黑电平的温漂及是否随模拟增益变化


**需要标定的模块**

- BLC
- WDR Expand
- Noise profile (NP)（噪声分布的均值标准差等）
- GE
- Dynamic dead pixel correction (Dynamic DPC)
- AE
- LSC
- AWB
- Gamma
- CCM
- Chromatic aberration correction (CAC)
- Purple fringe correction (PF)