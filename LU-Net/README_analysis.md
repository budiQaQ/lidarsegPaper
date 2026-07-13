# LU-Net 解读

## 一句话总结

LU-Net 是一个早期的 LiDAR range-image 语义分割方法：先基于 LiDAR sensor topology 为每个点学习局部 3D 几何特征，再把这些特征投影成多通道 range image，最后用 U-Net 做 2D 语义分割。

## 方法动机

PointNet 这类 raw point 方法可以直接处理点云，但对完整 LiDAR scan 来说计算和扩展性不理想；SqueezeSeg 等 range-image 方法很快，但输入通道通常只是 depth、reflectance 等低级特征，2D CNN 很难直接理解真实 3D 局部几何。LU-Net 的目标是在两者之间折中：保留 range-image 的高效 2D 处理，同时在投影前学习每个点的 3D 邻域特征。

## 网络结构与关键模块

- sensor topology: 利用 LiDAR 的扫描拓扑构造 range image，并估计每个点的 8-connected neighborhood。
- learned 3D feature extraction: 代码中 `pointnet=True` 时，对每个点的 3x3 patch 去掉中心点后得到 8 个邻居，输入 PointNet-like shared 1x1 conv + max pooling 模块，生成局部 embedding。
- relative coordinates: ablation 显示用相对邻域坐标比绝对坐标更好，因为能学习与全局位置无关的局部几何模式。
- range-image projection: 代码默认读取 `xyzdr` 五通道，其中邻居的 xyz 会减去中心点 xyz 变成局部相对坐标；PointNet-like 模块输出 3 通道 embedding，并与中心点的前 4 个通道拼接后继续卷积。
- U-Net segmentation: 使用简单 U-Net encoder-decoder 做最终逐像素分割，再映射回 3D 点。
- focal loss: 用于缓解类别不均衡，尤其对 pedestrian / cyclist 等少样本类别有帮助。

## 输入/输出与表示

LU-Net 的输入来自 LiDAR point cloud，但不是直接对无序点集做全局 PointNet，也不是只用原始 depth/reflectance range image。它先在 3D 点邻域中学习局部特征，再将学习后的特征作为 U-Net 的 2D 多通道输入。因此它是 hybrid 方法：前端偏 3D local feature，后端偏 projection-based image segmentation。

## Loss / Training / Evaluation

- 论文使用 IoU 作为主要评估指标。
- 代码中 `u_net_loss` 支持 focal loss，默认 `focal_loss=True`，并叠加基于边界距离的像素权重 `weights_ce = 0.1 + exp(-dist / (2 * 3^2))`，对目标边界附近像素给更高权重。
- 在 KITTI 数据集上评估 car、pedestrian、cyclist 三类。
- 论文报告 LU-Net 在单 GPU 上约 24 FPS。
- README 说明需下载 SqueezeSeg 数据集，并用 `make_tfrecord_pn.py` 生成训练/验证 TFRecords。

## 实验数据集与指标

论文 Table 1 在 KITTI 上对比 PointSeg、SqueezeSeg、SqueezeSegV2、RIU-Net：

- SqueezeSeg: average IoU 37.2
- PointSeg: average IoU 39.8
- RIU-Net: average IoU 40.6
- SqueezeSegV2: average IoU 44.9
- LU-Net: average IoU 55.4

其中 LU-Net 在 pedestrian 和 cyclist 上提升明显，分别达到 46.9 和 46.5 IoU。论文认为主要原因是学习到的 3D 局部特征能补足只用 depth/reflectance range image 的不足。

## 优点

- 思路清晰：用 learned 3D feature 弥补 range-image 2D CNN 的几何信息不足。
- 比纯 raw point 方法更适合实时 LiDAR scan，论文报告 24 FPS。
- 比早期 SqueezeSeg / RIU-Net 更关注小目标类别，如 pedestrian、cyclist。
- U-Net 后端简单直观，易理解。
- 作为 SalsaNext、FIDNet、CENet 之前的 projection-based 过渡方法，很有谱系价值。

## 缺点

- 有 TensorFlow 实现可参考，但代码年代较早，且依赖 SqueezeSeg 数据集格式和手动路径配置。
- 论文使用的是 KITTI 早期三类设置，不是后续 SemanticKITTI 20 类标准，不能直接与 SalsaNext/FIDNet/CENet 的 mIoU 横比。
- 3D feature extraction module 需要构造邻域，工程复杂度高于直接输入 range image。
- 相比后续 SalsaNext/CENet，网络设计较早，缺少更成熟的 Lovasz loss、强上下文模块、辅助监督或不确定性估计。
- 依赖 sensor topology，对不同 LiDAR 线数和扫描模式的泛化需要额外验证。

## 工程复现风险

- 代码基于 TensorFlow 1.6，环境较旧。
- `semantic_base` 数据集路径写在 `make_tfrecord_pn.py` 中，不在 cfg 里，需要手动修改。
- 依赖 SqueezeSeg 数据集格式，而不是直接读取原始 KITTI / SemanticKITTI。
- 默认 `n_classes=4`，对应 background + car/pedestrian/cyclist 这类早期设置；迁移到 SemanticKITTI 20 类需要改数据映射和训练配置。
- 若迁移到 SemanticKITTI，需要重新设计类别映射和输出类别数。

## 适合借鉴的实现点

- 在 range-image CNN 前加入 learned 3D local feature。
- 用 relative neighbor coordinates 表达局部几何。
- 使用 3x3 range-image patch 的 8 邻居作为局部 PointNet-like feature extractor 输入。
- 保留 U-Net 这种简单 encoder-decoder 后端。
- 对少样本交通参与者类别使用 focal loss，并对边界附近像素加权。

## 与后续论文对比时应关注的维度

- 与 PointNet/PointNet++: LU-Net 不直接处理全局 raw point set，而是只学习局部 3D 邻域特征，再转 2D。
- 与 SalsaNext: 二者都用 encoder-decoder range image；LU-Net 的特色是投影前 3D feature，SalsaNext 的特色是更强 2D context/dilated encoder 和 uncertainty。
- 与 FIDNet/CENet: LU-Net 使用 U-Net decoder；FIDNet/CENet 更强调简洁/无参数插值解码和高 FPS。
- 与 CENet: CENet 后续用 3x3 conv 和 auxiliary loss 强化 range-image 表达，LU-Net 则从输入特征侧补 3D 几何。

## 参数量与计算耗时

- 参数量: 论文/README 未给出固定参数量；官方代码 `train.py` 会打印 TensorFlow trainable variables 总数，但本地未运行 TF1 环境。
- 计算耗时: 论文报告单 GPU 约 24 FPS，约 41.7 ms/scan；注意这是早期 KITTI 三类设置，不是 SemanticKITTI 20 类口径。

## 论文与代码地址

- 论文地址: https://arxiv.org/abs/1908.11656
- GitHub 仓库: https://github.com/budiQaQ/lunet

## 核心创新代码块

```python
# LU-Net 思想伪代码
neighbors = gather_8_neighbors_from_range_patch(points_xyz)
relative_xyz = neighbors - center_point
local_feature = shared_mlp(relative_xyz).max(dim="neighbors")
range_tensor = concat([xyz, depth, remission, local_feature])
logits = unet(range_tensor)
```

## 使用方法描述

使用时先生成带 3D 局部特征的 TFRecords，再训练 U-Net；默认配置依赖 TensorFlow 1.6 和 SqueezeSeg 数据组织。

