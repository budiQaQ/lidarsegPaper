# PointNet++ 元信息

- 论文标题: PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
- 作者: Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas
- 会议/年份: NeurIPS 2017
- arXiv: https://arxiv.org/abs/1706.02413
- 本地论文: `paper.pdf`
- 官方代码: https://github.com/charlesq34/pointnet2
- 本地代码: `code/`
- 代码 commit: `42926632a3c33461aebfbee2d829098b30a23aaa`
- 任务: point cloud classification, part segmentation, semantic scene parsing
- 方法类型: raw point-based hierarchical neural network
- 主要数据集: ModelNet40, ShapeNetPart, ScanNet
- 核心模块: set abstraction, farthest point sampling, ball query, local PointNet, multi-scale grouping, feature propagation
- 参数量与计算耗时: PointNet++ 的参数量和耗时强依赖 SSG/MSG/MRG 配置及点数，需按具体配置统计。
- 复现备注: 官方代码基于 TensorFlow 1.2，包含自定义 TF ops，需要编译 sampling/grouping/interpolation 等 CUDA 算子；未下载数据集。
