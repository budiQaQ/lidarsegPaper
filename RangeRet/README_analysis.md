# RangeRet 解读

## 一句话总结

RangeRet 是一个轻量 range-view LiDAR 分割方法，把 NLP 中的 Retentive Network 改造成适配 range image 的 Circular Retention，用较少参数建模长程上下文和 360 度水平环绕连续性。

## 方法动机

Range-view 方法速度快、部署友好，但普通 CNN 长程建模能力有限，ViT/RangeViT/RangeFormer 虽能增强全局上下文，却通常参数更多、attention 代价更高。RangeRet 选择 Retentive Network 作为替代注意力机制，希望以更轻量的方式获得长程依赖建模能力，并显式利用 range image 的水平环绕性质。

## 网络结构与关键模块

- Range Embedding / ConvStem 将 5 通道 range image 特征编码到高维特征。
- VisionEmbedding 将 range image 切成 patch tokens。
- RetNet backbone 替代 ViT self-attention，默认 8 层、4 heads、model dim 128。
- Circular Retention (CiR) 使用 circular distance 处理水平方向 360 度连续性；代码中 `col_diff = min(col_diff, W - col_diff)` 明确体现了环绕距离。
- SemanticHead 将 token 特征插值回原始 range image 分辨率，并与 stem residual 融合。
- 推理后可使用 RangeNet++ 风格 kNN post-processing。

## 输入/输出与投影表示

- 输入是 spherical range image，SemanticKITTI 配置默认 `64 x 1024`。
- 输入通道为 5 维：range, x, y, z, signal。
- 默认 patch size 为 `[7, 7]`，stride 为 `[4, 4]`，因此使用重叠 patch。
- 输出是 range-view 语义预测，再回投影到 3D 点。

## Loss / post-processing / training strategy

- 论文使用 cross entropy、Lovasz-Softmax 和 boundary loss。
- 训练使用 AdamW、WarmupCosine scheduler、stochastic depth，默认 64 epochs。
- 引入一组 range-view augmentations，由 3D augmentation 改造而来，用于提升泛化和缓解类别不均衡。
- evaluation 使用 kNN post-processing。

## 实验数据集与指标

- SemanticKITTI test: 64.5 mIoU。
- PandaSet test: 60.0 mIoU，在 range-view 方法中显著优于 CENet 55.4、FIDNet 51.1。
- SemanticPOSS test: 52.8 mIoU。
- PandaSet validation 上参数约 3.8M、runtime 38 ms；论文表中 SalsaNext 6.7M/19ms，FIDNet 6.0M/21ms，RangeFormer 23.7M/54ms。

## 优点

- 比 RangeFormer 轻很多，更适合实时和车端部署。
- CiR 对 range image 水平方向环绕连续性建模明确，是很适合 range-view 的 inductive bias。
- 不依赖外部 image pretraining，也能达到接近 CENet/RangeViT 的 SemanticKITTI 表现。
- 支持 SemanticKITTI、PandaSet、SemanticPOSS 三个数据集，工程覆盖面比很多 range-view 论文更广。

## 缺点

- SemanticKITTI test 64.5 mIoU，低于 RangeFormer 69.5/73.3 级别结果。
- 相比 FIDNet/CENet，结构引入 RetNet/CiR，理解和调参成本更高。
- 依然是 range-view 方法，projection aliasing 和遮挡问题没有完全消失。
- kNN post-processing 仍然存在，端到端点级恢复能力有限。

## 工程复现风险

- 论文/代码较新，需要关注依赖版本和后续修复。
- 默认配置使用 `64 x 1024`，与很多方法的 `64 x 2048` 结果对比时要看公平设置。
- PandaSet 需要额外 raw Pandar64 LiDAR 数据，并需转换到 SemanticKITTI 格式。
- 如果启用 fp16、不同 patch size 或不同 stride，速度/精度会明显变化。

## 适合借鉴的实现点

- 在 CENet/FIDNet bottleneck 中加入轻量 Retention/CiR block，而不是直接上重 ViT。
- LC 蒸馏中可以用 CiR 的 circular distance mask 约束 window/token KD，避免水平边界断裂。
- 用 overlapping `7 x 7 / stride 4` patch 作为 range-view tokenization 的轻量选择。
- 把 range-view augmentations 纳入 student 训练，和 camera teacher KD 做组合 ablation。

## 与过往论文对比时应关注的维度

- 与 RangeViT: RangeViT 用 ViT 和图像预训练；RangeRet 用 RetNet/CiR，更轻且显式建模 range image 环绕。
- 与 RangeFormer: RangeFormer 精度更高、框架更完整；RangeRet 参数更少、速度更友好。
- 与 CENet/FIDNet: RangeRet 的核心增益来自长程 retention 和 circular prior，而不是 decoder/loss 小改。
- 与 2DPASS: 2DPASS 是训练期图像先验；RangeRet 是纯 LiDAR range-view student，可作为 LC 蒸馏的轻量 student backbone。
