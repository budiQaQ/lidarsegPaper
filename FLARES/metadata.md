# FLARES Metadata

- 论文标题: FLARES: Fast and Accurate LiDAR Multi-Range Semantic Segmentation
- 作者: Bin Yang, Alexandru Paul Condurache
- 会议/年份: WACV 2026 / arXiv 2025
- arXiv: https://arxiv.org/abs/2502.09274
- 项目页: https://binyang97.github.io/FLARES/
- 本地论文: `paper.pdf`
- 代码: 项目页标注 Code coming soon；截至本次整理未发现官方公开 GitHub 仓库
- 本地代码: 未下载
- 任务: LiDAR point cloud semantic segmentation
- 主要数据集: SemanticKITTI, nuScenes
- 方法类型: range-view / multi-range training paradigm / projection-based LiDAR segmentation
- 核心模块: spherical-coordinate point cloud splitting, low-resolution multi-range projection, WPD+ augmentation, Multi-Cloud Fusion, NNRI post-processing
- 参数量与计算耗时: FLARES 是训练/推理范式而非单一 backbone；论文表中增强后的 SalsaNext 约 6.7M、29 ms，FIDNet 约 6.0M、19 ms，CENet 约 6.8M、22 ms，RangeViT 约 24.1M、29 ms，并报告相对 baseline 超过 40% inference speed-up。
- 代表结果:
  - SemanticKITTI: 对 SalsaNext / FIDNet / CENet / RangeViT 分别带来约 5.3 / 7.9 / 3.3 / 2.1 mIoU 提升。
  - nuScenes: 对 SalsaNext / FIDNet / CENet / RangeViT 分别带来约 2.3 / 3.9 / 3.1 / 1.8 mIoU 提升。
  - 论文摘要报告 FLARES 在两个数据集和多种 architecture 上提升 mIoU，并带来超过 40% inference speed-up。
- 复现备注: 论文代码尚未公开；短期只能参考论文伪代码和已公开的 FIDNet/CENet/SalsaNext/RangeViT 仓库重实现。不要默认下载 SemanticKITTI、nuScenes 或 Carla synthetic data。
