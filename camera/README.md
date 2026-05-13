# Camera

这里放以相机成像链路为中心的知识：光学、sensor、模组、标定、ISP、IQ、RAW/RGB 图像处理、低光、降噪、HDR、TOF、Neural ISP。

归属原则：问题域优先，方法其次。只要核心问题离不开相机链路、RAW、sensor、ISP pipeline 或 IQ 调试，即使用了神经网络，也优先放在这里；通用网络结构、loss、backbone 和训练范式放到 `machine learning`。

常见边界：
- `ISP`：传统 ISP 模块、3A、标定、调试策略。
- `denoise`：噪声模型、去噪训练范式、RGB/RAW 去噪问题。
- `raw_denoise`：RAW 域噪声、VST、k-sigma、YOND、PMRID 等。
- `low_light`：曝光不足、可见性增强、Retinex、夜间感知任务。
- `nn_isp`：learned ISP、neural ISP、相机链路里的网络化模块。
- `restoration`：和相机/RAW/多帧成像强相关的复原、超分、视频质量增强。