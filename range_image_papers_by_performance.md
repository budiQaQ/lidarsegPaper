# Range-Image LiDAR Segmentation 论文性能整理

## 排序口径

优先按 SemanticKITTI single-scan semantic segmentation 的公开 mIoU 表现排序；如果论文只在早期 KITTI car/pedestrian/cyclist 三类设置上评估，则单独放在“不可直接横比”部分。

注意：不同论文的输入分辨率、是否计入 post-processing、硬件、test/val split、是否使用额外数据或蒸馏都可能不同。下面排序用于确定调研优先级，不等价于严格 benchmark 复现结论。

## 已整理论文中的 Range-Image / Projection 方法

| 性能层级 | 论文 | 已整理 | 主要指标/表现 | 参数量 | 计算耗时/速度 | 方法定位 |
| --- | --- | --- | --- | --- | --- | --- |
| 高 | CENet | 是 | SemanticKITTI test 约 64.7 mIoU | 代码可统计，当前未运行 PyTorch | 64x2048 约 37.79 FPS；64x1024 约 67.97 FPS；64x512 约 84.91 FPS | FIDNet 思路上的增强版，3x3 conv + activation + auxiliary heads |
| 高 | RangeRet | 是 | SemanticKITTI test 64.5 mIoU；PandaSet test 60.0；SemanticPOSS test 52.8 | 约 3.8M | PandaSet validation 约 38 ms | Retentive Network + Circular Retention，轻量 range-view 长程建模 |
| 高 | RangeViT | 是 | SemanticKITTI test 64.0 mIoU；nuScenes val 75.2 mIoU | 未确认，取决于 ViT backbone | 未找到可横比 latency | range image + image-pretrained ViT + 3D refiner |
| 高 | Lite-HDSeg | 是 | SemanticKITTI test 63.8 mIoU，约 20 FPS | 未确认 | 约 20 FPS，约 50 ms/scan | lite harmonic dense convolution + ICM + MCSPN，高精度实时 encoder-decoder |
| 高 | KPRNet | 是 | SemanticKITTI test 63.1 mIoU | 未确认 | 对比表约 0.3 FPS，约 3333 ms/scan | 2D projection CNN + KPConv learnable point-wise refinement |
| 中高 | SalsaNext | 是 | SemanticKITTI test 约 59.5 mIoU | 后续对比表约 6.7M | 后续对比表约 19 ms | 完整 range-image encoder-decoder，dilated context + pixel shuffle + uncertainty |
| 中高 | FIDNet | 是 | validation 约 58.8/58.9 mIoU；补充结构约 60 mIoU | 约 6M | CNN-only 约 11 ms；NLA 约 1.2 ms；FP16 单帧约 0.01 s | 极简 parameter-free interpolation decoding + NLA |
| 中 | SqueezeSegV3 | 是 | SqueezeSegV3-53+kNN 55.9 mIoU | 未确认 | 约 6 scans/sec，约 167 ms/scan | spatially-adaptive convolution，针对 range image 空间非平稳性 |
| 中 | 3D-MiniNet | 是 | 3D-MiniNet-KNN 55.8 mIoU；small 版 51.8 mIoU | small 1.13M；标准版 3.97M | small 61 FPS；标准版 36 FPS | learned 2D representation + 轻量 2D FCNN |
| 中 | RangeNet++ | 是 | RangeNet53++ 52.2 mIoU，约 12 scans/sec | 未确认 | 约 12 scans/sec，约 83 ms/scan | 经典 spherical projection + Darknet + kNN baseline |
| 早期不可横比 | LU-Net | 是 | KITTI 三类 average IoU 55.4，24 FPS | 未报告，代码可统计 | 约 24 FPS，约 41.7 ms/scan | learned 3D local feature + U-Net，非 SemanticKITTI 20 类 |

## Range-View Transformer 强基线

| 论文 | 已整理 | 主要指标/表现 | 参数量 | 计算耗时/速度 | 方法定位 |
| --- | --- | --- | --- | --- | --- |
| RangeFormer | 是 | SemanticKITTI test 73.3 mIoU；STR 低分辨率训练 72.2 mIoU | 约 23.7M | PandaSet validation 约 54 ms | 当前最重要的 LiDAR-only range-view Transformer/full-cycle framework 参考 |
| RangeRet | 是 | SemanticKITTI test 64.5 mIoU；PandaSet test 60.0；SemanticPOSS test 52.8 | 约 3.8M | PandaSet validation 约 38 ms | 比 RangeFormer 更轻，适合部署友好的 range-view Transformer/RetNet student |

## 仍建议后续补齐的谱系论文

| 优先级 | 论文 | 代表表现 | 为什么值得补 |
| --- | --- | --- | --- |
| 1 | MINet: Multi-scale Interaction Network for Real-time LiDAR Data Segmentation | 强调实时，多次出现在后续对比表 | 多尺度交互和嵌入式实时性，可作为轻量实时路线补充 |
| 2 | SqueezeSegV2 | KITTI 三类/LU-Net 对比中 average IoU 44.9 | SqueezeSeg 的增强版，LU-Net 直接对比对象 |
| 3 | RIU-Net | LU-Net 论文中 average IoU 40.6 | LU-Net 直接前身：range image + U-Net，但没有 learned 3D local feature |
| 4 | SqueezeSeg | LU-Net 论文中 average IoU 37.2；论文报告 8.7 ms/frame | 早期 range-image + CNN + recurrent CRF 起点，谱系价值高 |

## 非 Range-Image 但必须作为性能参照的方法

| 论文 | 已整理 | 主要指标/表现 | 参数量 | 计算耗时/速度 | 为什么单独列出 |
| --- | --- | --- | --- | --- | --- |
| 2DPASS | 是 | SemanticKITTI single-scan test 72.9 mIoU；训练期用图像，推理期只用 LiDAR | 1.9M 小模型；45.6M 大模型 | 论文表约 62 ms | 它不是 range-image 方法，而是 sparse point-voxel 3D backbone + 2D priors KD；性能显著高于已整理 range-image 组，适合作精度上限参照 |
| Point Transformer V3 | 是 | nuScenes val 80.3 mIoU；Waymo semantic 71.2；论文图中 SemanticKITTI 63.5 | 未确认 | 整理中记录约 44 ms | 不是 range-image，适合作 3D Transformer teacher/upper-bound 对照 |

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
