# Stratified Transformer 解读

## 一句话总结

Stratified Transformer 用“近处密集、远处稀疏”的 key sampling 在点云 Transformer 中低成本扩大感受野，是长程上下文建模的重要参考。

## 方法动机

3D 点云分割需要局部细节和远程上下文。只聚合局部邻域容易缺少全局结构；全局 attention 又太贵。Stratified Transformer 通过分层采样 key，让每个 query 同时看到近处密集点和远处稀疏点。

## 网络结构与关键模块

- 将空间划分为 non-overlapping cubic windows。
- 对每个 query，在本地窗口取 dense keys。
- 同时通过 FPS 下采样后，在更大窗口取 sparse distant keys。
- dense + sparse keys 合并后做标准 multi-head self-attention。
- 代码实现了 memory-efficient variant-length tokens 和 CUDA kernels。

## 输入/输出与表示

- 输入是 3D point cloud。
- 输出是点级语义标签。
- 原论文主要评估室内 S3DIS / ScanNetv2，不是 SemanticKITTI。

## Loss / post-processing / training strategy

- 论文重点是 Transformer block 和 key sampling。
- 代码提供 S3DIS 和 ScanNetv2 的训练/测试配置。
- 需要编译 `lib/pointops2`。

## 实验数据集与指标

- README 表述在 S3DIS 和 ScanNetv2 上达到当时 SOTA。
- 论文重点对比点式 Transformer 与 voxel-based 方法，强调 point-based 方法首次超过 voxel-based ones。

## 优点

- 分层 key sampling 对 range-view 很有启发：近距离/边界位置密集建模，远距离上下文稀疏建模。
- 比全局 attention 更可控。
- 对 LC 蒸馏中的“局部边界 + 远程语义一致性”设计有参考价值。

## 缺点

- 主要是室内点云方法，不能直接迁移到 LiDAR range image。
- 自定义 CUDA kernel 和 pointops2 增加复现/部署成本。
- 与车载实时 range-view pipeline 的工程目标不同。

## 工程复现风险

- 依赖 CUDA 编译，macOS 本地无法直接训练。
- S3DIS 结果有论文 README 提到的 ±0.5 mIoU 波动。
- 若用作自动驾驶对比，需要另行适配 SemanticKITTI/nuScenes。

## 适合借鉴的实现点

- 在 range-view Transformer 中设计 “local dense + global sparse” attention。
- 对 LC 蒸馏，边界/小目标区域用密集 window KD，远距离区域用稀疏 global token KD。
- 作为 3D point Transformer teacher 的结构参考。

## 与过往论文对比时应关注的维度

- 与 Swin: 都避免全局 attention，Swin 用规则 shifted windows，Stratified 用点云空间分层采样。
- 与 RangeFormer/RangeViT: Stratified 不是 range-view 方法，但长程上下文思想可迁移。
- 与 PTv3: Stratified 更早，强调 key sampling；PTv3 更强调序列化、简化和速度。
