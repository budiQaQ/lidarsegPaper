# SqueezeSegV3 解读

## 一句话总结

SqueezeSegV3 针对 LiDAR range image 的空间分布非平稳问题提出 Spatially-Adaptive Convolution，让卷积核随图像位置/输入自适应变化。

## 方法动机

普通 RGB 图像中，同一类局部模式可在图像不同位置共享卷积核；LiDAR range image 则不同，上下行对应不同俯仰角，近远距离、地面/天空区域分布差异明显。标准卷积强行共享权重会浪费模型容量，因此论文提出 SAC 让卷积具备位置自适应能力。

## 网络结构与关键模块

- SAC 通过 attention map 对标准卷积权重进行空间自适应调制。
- 代码中 `src/backbones/SAC.py` 包含 `SACBlock` 和 backbone，整体框架沿用 RangeNet++ 的训练、投影、kNN 后处理代码结构。
- 网络有 SqueezeSegV3-21 和 SqueezeSegV3-53 两个复杂度版本，对应 RangeNet21/53 级别。

## 输入/输出与投影表示

- 输入为 SemanticKITTI scan 投影成 64 x 2048 range image。
- 输入通道包含 range、xyz、remission 等配置项。
- 输出为 range-image 语义图，再通过可选 kNN post-processing 回填到点云。

## Loss / post-processing / training strategy

- 使用 multi-layer cross entropy loss，对多个 stage 输出施加监督。
- 类别权重按频率归一。
- 可使用 RangeNet++ 的 kNN post-processing，论文表中 `*` 表示启用 kNN。

## 实验数据集与指标

- SqueezeSegV3-53: 52.9 mIoU。
- SqueezeSegV3-53+kNN: 55.9 mIoU，约 6 scans/sec。
- 相比 RangeNet53+kNN 52.2 mIoU，提升主要来自 SAC 对 LiDAR range image 空间分布差异的建模。

## 优点

- 问题定义很清楚：不是简单堆 backbone，而是针对 range image 的非平稳特性改卷积算子。
- 代码开源且基于 RangeNet++，便于和 RangeNet++、SalsaNext、CENet 做结构对照。
- SAC 思想可迁移到其他 range-image backbone。

## 缺点

- SAC 增加额外参数和 MAC，尤其复杂版本速度下降明显。
- 最终点级恢复仍依赖 kNN，边界恢复没有端到端学习。
- 代码环境较旧，PyTorch 1.1.0 复现需要环境隔离。

## 工程复现风险

- SAC 的 attention 维度和具体变体会影响速度/显存，需要按配置核对。
- kNN post-processing 是否开启会显著影响 mIoU 和速度。
- 旧版依赖可能与当前 CUDA/PyTorch 不兼容。

## 适合借鉴的实现点

- 在 CENet 这类简洁 backbone 中尝试局部位置自适应卷积。
- 用 RangeNet++ 框架快速替换 backbone 做公平比较。
- 对 LiDAR range image 显式考虑 vertical row / horizontal position 的分布差异。

## 与后续论文对比时应关注的维度

- 与 RangeNet++: 同框架下标准卷积 vs SAC。
- 与 CENet: 空间自适应卷积 vs 普通 3x3 conv + activation/auxiliary loss。
- 与 Lite-HDSeg: SAC 算子改造 vs harmonic dense + ICM + MCSPN 系统改造。
- 与 KPRNet: 2D 自适应卷积 vs 3D 点级可学习 refinement。
