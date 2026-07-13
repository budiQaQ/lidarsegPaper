# PointNet 元信息

- 论文标题: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
- 作者: Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
- 会议/年份: CVPR 2017
- arXiv: https://arxiv.org/abs/1612.00593
- 本地论文: `paper.pdf`
- 官方代码: https://github.com/charlesq34/pointnet
- 本地代码: `code/`
- 代码 commit: `2618f72bc1a0fd21b074096e748016960d44ef55`
- 任务: point cloud classification, part segmentation, indoor semantic segmentation
- 方法类型: raw point-based neural network
- 主要数据集: ModelNet40, ShapeNetPart, S3DIS
- 核心模块: shared MLP, symmetric max pooling, input transform T-Net, feature transform T-Net, global feature + point feature fusion
- 参数量与计算耗时: PointNet 原始任务不是 SemanticKITTI LiDAR scan segmentation，参数量和耗时需按具体分类/分割配置本地统计。
- 复现备注: 官方代码基于 Python 2.7、TensorFlow 1.0.1、CUDA 8.0；未下载 ModelNet40、ShapeNetPart 或 S3DIS 数据。
