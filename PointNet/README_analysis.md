# PointNet 解读

## 一句话总结

PointNet 是直接处理无序点集的开山方法，用 shared MLP 提取逐点特征，再用对称 max pooling 得到全局特征，从结构上保证点顺序不影响输出。

## 方法动机

在 PointNet 之前，点云通常被转成 voxel、multi-view image 或 handcrafted feature，这会引入体素量化、视角选择和计算膨胀问题。PointNet 的核心观点是：点云本质上是 unordered set，网络需要直接尊重点集的 permutation invariance，而不是强行转成规则网格。

## 网络结构与关键模块

- shared MLP: 对每个点独立使用同一组 MLP/1x1 conv，提取 point-wise feature。
- symmetric function: 使用 max pooling 聚合所有点，获得 permutation-invariant global feature。
- T-Net: 学习输入点坐标变换和特征空间变换，提升对刚体/仿射变化的鲁棒性。
- classification: global feature 后接 fully connected layers 输出类别。
- segmentation: 将 global feature tile 回每个点，与 point feature 拼接后逐点分类。
- 代码中 `models/pointnet_cls.py` 对应分类模型，`models/pointnet_seg.py` 对应 part segmentation。

## 输入/输出与表示

输入通常是 `N x 3` 点坐标，也可附加 normal 等特征。输出可以是整云类别，也可以是每个点的 part/semantic label。PointNet 不依赖投影，不需要体素化，因此没有 range-image many-to-one mapping 问题。

## Loss / Training / Evaluation

- 分类与分割主要使用 cross entropy。
- feature transform matrix 通过正交约束正则化，减少变换矩阵退化。
- 官方代码支持 ModelNet40 分类、ShapeNetPart part segmentation 和 S3DIS indoor semantic segmentation。
- 代码环境较老，主要是 TensorFlow 1.x。

## 优点

- 直接处理 raw points，不需要 projection 或 voxelization。
- permutation invariance 设计非常清晰，是后续大量点云网络的基础。
- 结构简单，计算效率高，适合小规模点集分类和 part segmentation。
- 理论上分析了 set function 近似能力、critical point set 和鲁棒性。

## 缺点

- 局部结构建模弱。每个点先独立编码，靠全局 max pooling 聚合，缺少显式邻域关系。
- 对大规模 LiDAR scene segmentation 不够自然，需要 block/采样处理，难以直接吃完整一帧高密度 LiDAR。
- 对细粒度边界和局部几何模式不如 PointNet++、KPConv、range-image CNN 等方法。
- 官方代码年代较早，复现依赖老 TensorFlow/CUDA。

## 工程复现风险

- TensorFlow 1.0.1、Python 2.7、CUDA 8.0 环境较旧。
- S3DIS/ShapeNetPart 数据预处理格式较固定，需要按官方脚本生成 HDF5。
- 如果迁移到 PyTorch 或现代 TF，需核对 T-Net、BN、dropout 和数据增强细节。

## 适合借鉴的实现点

- 用 symmetric pooling 处理无序集合。
- point-wise shared MLP 作为局部/逐点编码器。
- global feature 与 point feature 拼接做分割。
- T-Net 的输入/特征对齐思想。

## 与后续论文对比时应关注的维度

- 后续方法是否显式建模 local neighborhood。
- 是否能扩展到大规模户外 LiDAR。
- 是否保持 raw-point 表示，还是转向 projection/voxel。
- 速度瓶颈来自点数、邻域搜索还是网络本体。
