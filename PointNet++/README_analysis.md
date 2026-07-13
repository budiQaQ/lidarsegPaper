# PointNet++ 解读

## 一句话总结

PointNet++ 是 PointNet 的层次化扩展：通过 sampling + grouping 构造局部邻域，在每个局部区域内递归使用 PointNet，从而学习从局部几何到全局语义的多尺度特征。

## 方法动机

PointNet 能直接处理无序点集，但缺少局部结构建模。真实点云存在欧氏距离、局部邻域和非均匀采样密度，直接全局 max pooling 会忽略很多细粒度几何模式。PointNet++ 的目标是像 CNN 一样逐层扩大感受野，但保持 raw point set 表示。

## 网络结构与关键模块

- set abstraction: 由 sampling layer、grouping layer 和 local PointNet layer 组成。
- FPS: 使用 farthest point sampling 选择覆盖点云的 centroid。
- ball query: 根据半径构造局部邻域，而不是在规则网格上滑窗。
- local PointNet: 对每个局部邻域使用 PointNet 聚合局部特征。
- MSG/MRG: multi-scale grouping / multi-resolution grouping 用于处理非均匀点密度。
- feature propagation: 对 segmentation 任务，将稀疏高层特征插值回原始点，再通过 skip connection 恢复点级预测。

## 输入/输出与表示

输入仍是 raw points，通常为 `N x 3` 或 `N x C`。分类任务逐层抽象到全局 feature；语义分割任务通过 feature propagation 输出每个点类别。与 range-image 方法不同，PointNet++ 不依赖球面投影，因此不会产生投影遮挡和像素冲突，但需要邻域搜索和采样。

## Loss / Training / Evaluation

- 分类和分割主要使用 cross entropy。
- segmentation 支持 sample weight。
- 官方代码包含 ModelNet40、ShapeNetPart 和 ScanNet 示例。
- 需要编译自定义 TensorFlow ops，包括采样、分组和插值。

## 优点

- 明确补齐 PointNet 的局部结构短板。
- 层次化 receptive field 更适合复杂几何和 scene-level segmentation。
- MSG/MRG 对非均匀采样更鲁棒，适合真实传感器点云。
- feature propagation 思路对后续点云分割网络影响很大。

## 缺点

- FPS、ball query、grouping 和 interpolation 带来显著工程复杂度。
- 对大规模户外 LiDAR，邻域搜索和多层采样开销较高，不如 range-image CNN 容易实时化。
- 自定义 TF ops 增加编译和环境风险。
- 相比 projection-based 方法，硬件友好性和批处理效率较弱。

## 工程复现风险

- TensorFlow 1.2 及 CUDA 自定义算子环境较老。
- `tf_ops` 编译需要根据本地 TF include/library 路径修改脚本。
- 不同点数、半径、采样密度会显著影响效果。
- MSG/MRG 的多尺度参数需要按数据集调优。

## 适合借鉴的实现点

- set abstraction 作为 raw point 层次编码模板。
- FPS + ball query 构造局部 patch。
- local PointNet 学习 unordered local set feature。
- feature propagation + skip link 做点级分割。
- 多尺度邻域处理非均匀密度。

## 与后续论文对比时应关注的维度

- 后续方法是否继续使用 raw point neighborhood。
- 邻域搜索和采样开销是否计入总延迟。
- 对大规模 LiDAR 的扩展方式：block、random sampling、range projection 还是 sparse voxel。
- 是否在局部结构、实时性和部署复杂度之间取得更好平衡。

## 论文与代码地址

- 论文地址: https://arxiv.org/abs/1706.02413
- GitHub 仓库: https://github.com/charlesq34/pointnet2

## 核心创新代码块

```python
# PointNet++ set abstraction 核心伪代码
centroids = farthest_point_sample(points, npoint)
groups = query_ball_point(radius, nsample, points, centroids)
local_feat = pointnet_mlp(groups).max(dim="neighbors")
features = feature_propagation(centroids, points, local_feat)
```

## 使用方法描述

使用时需要编译官方 TF ops：sampling、grouping、interpolation；它更适合作 raw point 局部几何 baseline，不适合直接替代实时 range-view pipeline。

