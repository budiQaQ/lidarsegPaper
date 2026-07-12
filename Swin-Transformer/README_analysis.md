# Swin Transformer 解读

## 一句话总结

Swin Transformer 是通用视觉 backbone，通过 shifted window attention 让 Transformer 具备层次化、多尺度和较低计算成本，适合借鉴到超宽 range image 的局部窗口建模。

## 方法动机

标准 ViT 的全局 self-attention 对高分辨率图像代价高，也缺少 CNN 那种天然多尺度金字塔结构。Swin 用局部窗口 attention 控制复杂度，再通过 shifted window 让不同窗口之间交换信息。

## 网络结构与关键模块

- window multi-head self-attention 只在局部窗口内计算 attention。
- shifted window 在相邻层中移动窗口划分，建立跨窗口连接。
- patch merging 构成层次化 feature map，适合 dense prediction。
- 相对位置偏置增强局部空间建模。

## 输入/输出与表示

- 原论文面向 RGB 图像。
- 对 range-view LiDAR 可借鉴为 `H x W` range image 上的窗口 attention。
- 对 `64 x 2048` 这类超宽输入，建议使用横向长、纵向短的非方形窗口，而不是直接照搬 `7 x 7`。

## Loss / post-processing / training strategy

- 原论文主要是 backbone 训练与下游 fine-tuning。
- 对 LC 蒸馏更有价值的是结构设计，而不是原始分类/检测 loss。
- README 还链接了 Feature Distillation、SimMIM、SwinV2 等后续工作，可作为 teacher 预训练来源。

## 实验数据集与指标

- ImageNet-1K top-1 最高 87.3。
- COCO test-dev 58.7 box AP / 51.1 mask AP。
- ADE20K semantic segmentation 53.5 mIoU。

## 优点

- shifted window 非常适合控制 range-view Transformer 成本。
- 层次化输出便于做 multi-scale KD。
- 有大量图像预训练模型，可作为 camera teacher 或初始化 range-view encoder。

## 缺点

- 不是 LiDAR 方法，不能直接说明在 range image 上有效。
- 方形 window 不一定适合 LiDAR range image 的几何结构。
- 如果直接换 backbone，可能比 CENet/FIDNet 重很多。

## 工程复现风险

- 原仓库以图像分类为主，语义分割需要另一个官方 segmentation repo 或 MMDetection/MMSeg 配置。
- 对 range-view 需要改输入通道、window shape、positional encoding 和 mask。
- 部署时要关注 window partition / roll / attention 的 TensorRT 支持。

## 适合借鉴的实现点

- 在 CENet/FIDNet bottleneck 处插入轻量 shifted window block。
- LC 蒸馏中按 window 做 feature KD，而不是整图全局 L2。
- 用 Swin/SwinV2/SegFormer 类 image teacher 提供多尺度边界和语义先验。

## 与过往论文对比时应关注的维度

- 与 RangeViT: RangeViT 是 ViT 直接适配 range image；Swin 更适合作局部窗口 attention 设计灵感。
- 与 RangeFormer: RangeFormer 是 LiDAR range-view 完整方案；Swin 只是可借鉴模块。
- 与 2DPASS: Swin 可作为 2D camera branch teacher，不是 LiDAR student。
