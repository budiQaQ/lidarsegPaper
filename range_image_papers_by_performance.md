# Range-Image LiDAR Segmentation 论文性能整理

## 排序口径

优先按 SemanticKITTI single-scan semantic segmentation 的公开 mIoU 表现排序；如果论文只在早期 KITTI car/pedestrian/cyclist 三类设置上评估，则单独放在“不可直接横比”部分。

注意：不同论文的输入分辨率、是否计入 post-processing、硬件、test/val split、是否使用额外数据或蒸馏都可能不同。下面排序用于确定调研优先级，不等价于严格 benchmark 复现结论。

## 已整理论文中的 Range-Image / Projection 方法

| 性能层级 | 论文 | 已整理 | 主要指标/表现 | 方法定位 |
| --- | --- | --- | --- | --- |
| 高 | CENet | 是 | SemanticKITTI test 约 64.7 mIoU | FIDNet 思路上的增强版，3x3 conv + activation + auxiliary heads |
| 高 | Lite-HDSeg | 是 | SemanticKITTI test 63.8 mIoU，约 20 FPS | lite harmonic dense convolution + ICM + MCSPN，高精度实时 encoder-decoder |
| 高 | KPRNet | 是 | SemanticKITTI test 63.1 mIoU | 2D projection CNN + KPConv learnable point-wise refinement |
| 中高 | SalsaNext | 是 | SemanticKITTI test 约 59.5 mIoU | 完整 range-image encoder-decoder，dilated context + pixel shuffle + uncertainty |
| 中高 | FIDNet | 是 | validation 约 58.8/58.9 mIoU；补充结构约 60 mIoU | 极简 parameter-free interpolation decoding + NLA |
| 中 | SqueezeSegV3 | 是 | SqueezeSegV3-53+kNN 55.9 mIoU | spatially-adaptive convolution，针对 range image 空间非平稳性 |
| 中 | 3D-MiniNet | 是 | 3D-MiniNet-KNN 55.8 mIoU；small 版 51.8 mIoU/61 FPS | learned 2D representation + 轻量 2D FCNN |
| 中 | RangeNet++ | 是 | RangeNet53++ 52.2 mIoU，约 12 scans/sec | 经典 spherical projection + Darknet + kNN baseline |
| 早期不可横比 | LU-Net | 是 | KITTI 三类 average IoU 55.4，24 FPS | learned 3D local feature + U-Net，非 SemanticKITTI 20 类 |

## 仍建议后续补齐的谱系论文

| 优先级 | 论文 | 代表表现 | 为什么值得补 |
| --- | --- | --- | --- |
| 1 | MINet: Multi-scale Interaction Network for Real-time LiDAR Data Segmentation | 强调实时，多次出现在后续对比表 | 多尺度交互和嵌入式实时性，可作为轻量实时路线补充 |
| 2 | SqueezeSegV2 | KITTI 三类/LU-Net 对比中 average IoU 44.9 | SqueezeSeg 的增强版，LU-Net 直接对比对象 |
| 3 | RIU-Net | LU-Net 论文中 average IoU 40.6 | LU-Net 直接前身：range image + U-Net，但没有 learned 3D local feature |
| 4 | SqueezeSeg | LU-Net 论文中 average IoU 37.2；论文报告 8.7 ms/frame | 早期 range-image + CNN + recurrent CRF 起点，谱系价值高 |

## 非 Range-Image 但必须作为性能参照的方法

| 论文 | 已整理 | 主要指标/表现 | 为什么单独列出 |
| --- | --- | --- | --- |
| 2DPASS | 是 | SemanticKITTI single-scan test 72.9 mIoU；训练期用图像，推理期只用 LiDAR | 它不是 range-image 方法，而是 sparse point-voxel 3D backbone + 2D priors KD；性能显著高于已整理 range-image 组，适合作精度上限参照 |

## 性能排序带来的工程判断

- 追求最高 mIoU: CENet、Lite-HDSeg、KPRNet 是当前已整理 range/projection 组的前三。
- 追求轻量实时 baseline: FIDNet、CENet、RangeNet++ 更适合直接改造；3D-MiniNet 也适合做小模型对照。
- 追求边界/点级恢复: KPRNet 的 KPConv refinement、Lite-HDSeg 的 MCSPN、RangeNet++/SqueezeSegV3/3D-MiniNet 的 kNN、FIDNet 的 NLA 是四条不同路线。
- 追求卷积算子创新: SqueezeSegV3 的 SAC 和 Lite-HDSeg 的 harmonic dense convolution 最值得单独拆解。
- 追求谱系完整: RangeNet++ 是后续 SqueezeSegV3、3D-MiniNet、SalsaNext 等代码/流程的重要基线。
- 追求更高精度上限且允许训练期引入相机图像: 2DPASS 是比 range-image 组更强的参照，但不能作为纯 range-image 结构公平横比。

## 对比时重点看什么

- 后处理路线: RangeNet++ 的 kNN、FIDNet 的 NLA、KPRNet 的 learnable KPConv、Lite-HDSeg 的 MCSPN+kNN。
- decoder 路线: SalsaNext/LU-Net/Lite-HDSeg 的 encoder-decoder，FIDNet/CENet 的 interpolation decoding。
- 输入特征路线: LU-Net/3D-MiniNet 先学习 3D/2D representation，RangeNet++/SalsaNext/FIDNet/CENet/SqueezeSegV3 直接使用 range-image 多通道输入。
- 卷积设计路线: SqueezeSegV3 的 spatially-adaptive convolution，CENet 的 3x3 conv + activation，Lite-HDSeg 的 harmonic dense convolution。
- 实时性: 是否报告完整 pipeline FPS，是否包含投影、网络、后处理、投影回点云。
- 跨模态训练: 2DPASS 需要单独标注训练期是否使用图像、推理期是否只用 LiDAR、是否启用 TTA/instance augmentation。
