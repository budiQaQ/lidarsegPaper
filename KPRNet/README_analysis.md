# KPRNet 解读

## 一句话总结

KPRNet 是把强 2D projection CNN 与 KPConv 点级 refinement 结合的 range-image LiDAR 分割方法，用可学习点云算子替代 RangeNet++ 一类方法中的固定 kNN 后处理。

## 方法动机

Range-image 方法速度快、工程友好，但投影到 2D 后会丢失 3D 邻域几何，且回投影时常依赖固定 kNN/CRF 后处理。KPRNet 的核心判断是：2D CNN 适合做全局/局部图像特征抽取，但最终点级标签恢复应当让模型学习 3D 点间关系，而不是只用手工距离规则。

## 网络结构与关键模块

- 2D backbone 使用更强的 projection CNN；官方代码中可见 ResNeXt101/DeepLab 风格 backbone、ASPP、多尺度低层特征融合。
- KPConv refinement 接在 2D 特征之后：先用 `grid_sample` 将 2D 特征按点的投影坐标采样回点，再用 KPConv 基于 3D 点和邻居索引做点级特征更新。
- 最终分类头直接输出点级类别，因此它不是单纯“像素预测后再投影”，而是把最后一步改成可学习的 point-wise component。

## 输入/输出与投影表示

- 输入 LiDAR scan 先投影成 range image，论文使用 SemanticKITTI。
- 官方代码中能看到 depth / reflectivity 两通道 range image，以及点的投影坐标 `px/py`、原始 3D 坐标 `points_xyz`、kNN 邻居索引。
- 输出是原始点级语义标签。

## Loss / post-processing / training strategy

- 训练重点在 2D projection CNN 与 KPConv 点级头联合优化。
- 传统 kNN post-processing 被 KPConv refinement 替代，这是 KPRNet 与 RangeNet++、SqueezeSegV3+kNN、Lite-HDSeg+kNN 的最大区别。
- 论文中先用 KNN 后处理建立 baseline，再用 KPConv 替换后处理并报告验证集提升。

## 实验数据集与指标

- SemanticKITTI test: 63.1 mIoU。
- 论文表中对比 RangeNet++ 52.2、SqueezeSegV3 55.9、SalsaNext 59.5，KPRNet 在当时明显领先 projection-based 方法。
- 速度方面 KPRNet 明显慢于纯 range-image 方法，Lite-HDSeg 表中给出的 FPS 约 0.3，说明 KPConv refinement 带来较大计算代价。

## 优点

- 抓住了 projection-based 方法的关键痛点：2D 图像邻域不等于真实 3D 邻域。
- KPConv refinement 比固定 kNN/NLA 更有表达能力，尤其有利于边界、小目标和投影遮挡区域。
- 方法定位清晰，适合作为“2D range image backbone + 3D point refinement”的代表。

## 缺点

- 速度和部署成本是主要问题，点级 KPConv 会显著削弱 range-image 方法的实时优势。
- 依赖邻域构建和点级算子，工程复杂度高于 FIDNet/CENet/RangeNet++。
- 训练依赖较重，官方 README 写明论文结果使用 8 张 16GB GPU，总 batch size 24。

## 工程复现风险

- KPConv 邻域搜索、batch padding、点数变化会影响显存和速度。
- 官方权重和 Cityscapes 预训练权重在 Google Drive，本次按要求未下载大体量权重。
- 若用于实时部署，需要重点评估 KPConv 后处理是否成为瓶颈。

## 适合借鉴的实现点

- 用 2D CNN 提取 dense feature，再采样回点级空间。
- 用可学习点级 refinement 替代固定 kNN/NLA。
- 将投影坐标、原始 3D 坐标、邻居索引作为显式接口，使 2D/3D 两部分边界清晰。

## 与后续论文对比时应关注的维度

- 与 RangeNet++: KPConv refinement vs fixed kNN post-processing。
- 与 FIDNet: 可学习点级恢复 vs parameter-free interpolation + NLA。
- 与 CENet: 更强 3D refinement vs 更轻量 2D 表达增强。
- 与 Lite-HDSeg: 点级 refinement vs 2D encoder-decoder + MCSPN/kNN。
