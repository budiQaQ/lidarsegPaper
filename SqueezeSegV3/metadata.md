# SqueezeSegV3 元信息

- 论文标题: SqueezeSegV3: Spatially-Adaptive Convolution for Efficient Point-Cloud Segmentation
- 作者: Chenfeng Xu, Bichen Wu, Zining Wang, Wei Zhan, Peter Vajda, Kurt Keutzer, Masayoshi Tomizuka
- 会议/年份: ECCV 2020
- arXiv: https://arxiv.org/abs/2004.01803
- 本地论文: `paper.pdf`
- 代码: https://github.com/chenfengxu714/SqueezeSegV3
- 本地代码: `code/`
- 本地 commit: `d12c16312e4f634bcf844200b6de87bf841575ca`
- 任务: LiDAR semantic segmentation
- 主要数据集: SemanticKITTI
- 表示形式: spherical range image
- 核心模块: Spatially-Adaptive Convolution, RangeNet++ framework, multi-layer cross entropy, optional kNN post-processing
- 参数量与计算耗时: SqueezeSegV3-53+kNN 约 6 scans/sec；参数量需按 SAC/backbone 版本重新统计。
- 代表结果: SqueezeSegV3-53+kNN SemanticKITTI test 55.9 mIoU
- 复现备注: 代码基于 PyTorch 1.1.0 / Python 3.6；README 提供预训练模型入口，但本次未下载权重。
