### 🏷️ 图像分类与回归

这类任务是计算机视觉的基础，损失函数旨在衡量预测值（类别概率或连续值）与真实标签之间的差距。

|损失函数|类型|主要特点|
|---|---|---|
|**交叉熵损失**|分类|最常用的分类损失函数，衡量两个概率分布之间的差异，适用于多分类和二分类任务。[](https://developer.aliyun.com/article/1507761)[](https://developer.baidu.com/article/details/3330319)|
|**均方误差**|回归|计算预测值与真实值之差的平方和的均值，对异常值比较敏感，常用于像素值等连续变量的预测。[](https://developer.baidu.com/article/details/3330319)|

### 📦 目标检测

目标检测不仅要识别物体的类别，还要定位其边界框，因此损失函数通常由**分类损失**和**边界框回归损失**两部分组成。

|损失函数|类型|主要特点|
|---|---|---|
|**Smooth L1 Loss**|边界框回归|结合了L1和L2损失的优点，在误差较小时梯度更平滑，对离群点不敏感，是Faster R-CNN等经典检测器的标配。[](https://developer.baidu.com/article/details/3330319)|
|**IoU Loss 及其变体**|边界框回归|直接优化预测框与真实框的交并比（IoU）指标。其中**CIoU Loss**和**EIoU Loss**等变体进一步考虑了框的重叠面积、中心点距离和长宽比，收敛更快、精度更高。[](https://developer.aliyun.com/article/1438127)|
|**Focal Loss**|分类|为了解决正负样本极度不平衡的问题而设计，通过降低易分类样本的权重，让模型更专注于难分类的样本，在小目标检测中尤为有效。[](https://developer.aliyun.com/article/1507761)[](https://developer.aliyun.com/article/1438127)|

### 🎨 图像分割

图像分割是像素级的分类任务，需要为图像中的每个像素分配一个类别标签。

|损失函数|类型|主要特点|
|---|---|---|
|**Dice Loss**|基于重合度|直接优化分割任务中常用的Dice系数（衡量两个集合相似度的指标），能有效缓解类别不均衡问题，特别适用于前景占比较小的医学图像分割。[](https://developer.aliyun.com/article/1507761)[](https://cloud.tencent.cn/developer/article/1589583)|
|**加权交叉熵**|基于交叉熵|为标准交叉熵损失中的每个类别分配不同的权重，以此来削弱数据集中像素数量占主导的类别（如背景）对模型的影响。[](https://developer.aliyun.com/article/1507761)[](https://cloud.tencent.cn/developer/article/1589583)|

### 🖼️ 图像生成

图像生成的任务是创造新的、逼真的图像，其损失函数旨在衡量生成图像的质量、多样性与目标分布的相似度。

| 损失函数           | 类型    | 主要特点                                                                                                                                                                                                                        |
| -------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **对抗损失**       | GAN核心 | 通过生成器和判别器的博弈来训练模型，推动生成器产生足以“以假乱真”的图像，是GAN的基础。[](https://www.sciencedirect.com/science/article/abs/pii/S0031320321005318?dgcid=rss_sd_all)[](https://pmc.ncbi.nlm.nih.gov/articles/PMC8164834/table/Tab5/)                   |
| **感知损失**       | 特征重建  | 不直接比较像素值，而是比较生成图像和真实图像在预训练网络（如VGG）高层特征图上的差异，能更好地捕捉图像的语义信息和视觉风格。[](https://www.sciencedirect.com/science/article/abs/pii/S0031320321005318?dgcid=rss_sd_all)[](https://pmc.ncbi.nlm.nih.gov/articles/PMC8164834/table/Tab5/) |
| **L1/L2 Loss** | 像素重建  | 像素级的重建损失，L1 Loss有助于保留图像亮度，L2 Loss则倾向于产生更平滑的结果，但可能导致图像模糊。通常作为辅助损失来约束图像的整体内容。                                                                                                                                                 |

### 📐 图像重建与增强领域的核心Loss分类

|损失类别|核心思想|代表Loss|关键特点 / 应用场景|
|---|---|---|---|
|**像素级损失**|直接计算预测图与真实图像素值的差异。|**L1 Loss** (MAE)  <br>**L2 Loss** (MSE)  <br>**Charbonnier Loss**|**基础且通用**。L1鼓励稀疏性，能更好地保留边缘；L2惩罚大误差，但易产生模糊结果。Charbonnier是L1的平滑变体，对离群点更鲁棒。[](https://pypi.org/project/sensecraft/)[](https://www.e-com-net.com/article/1643238636348563456.htm)|
|**结构相似性损失**|模仿人眼感知，从**亮度、对比度、结构**三方面评估图像相似度。|**SSIM Loss**  <br>**MS-SSIM Loss**|**追求视觉质量**。比像素级Loss更能体现图像的结构信息，生成结果更符合人眼观感。**MS-SSIM**通过多尺度计算，效果更稳定，常用于SR、去噪。[](https://pypi.org/project/sensecraft/)[](https://blog.csdn.net/qq_43275608/article/details/144852382)[](https://www.e-com-net.com/article/1643238636348563456.htm)|
|**梯度/边缘损失**|通过约束图像的**梯度信息**（即像素值的变化率）来保持边缘和纹理。|**Gradient Loss**  <br>**Sobel / Laplacian Loss**|**专注于清晰度**。能有效锐化边缘、抑制伪影，在图像超分、去模糊和医学图像分割中很关键，让生成的轮廓更分明。[](https://pypi.org/project/sensecraft/)[](https://blog.csdn.net/qq_43275608/article/details/144854032)[](https://blog.csdn.net/weixin_58349913/article/details/144843710)|
|**感知损失**|比较两图在预训练网络（如VGG）高层特征图上的差异，关注**语义级**相似。|**Perceptual Loss**  <br>**LPIPS**  <br>**DINO Loss**|**追求真实感**。不纠结像素是否一一对应，而是让图像在“风格”和“内容”上更接近，是**风格迁移和GAN生成**的核心驱动。[](https://pypi.org/project/sensecraft/)|
|**频域损失**|将图像变换到频域（如傅里叶变换），比较其**频率分量**的差异。|**FFT Loss**  <br>**Patch FFT Loss**|**关注纹理细节**。有助于重建图像的高频信息（如纹理），避免生成结果过于平滑，常用于修复纹理复杂的图像。|