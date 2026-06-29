# SalsaNext 元信息

- 论文标题: SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving
- 作者: Tiago Cortinhal, George Tzelepis, Eren Erdal Aksoy
- 会议/年份: ISVC 2020 / arXiv 2020
- arXiv: https://arxiv.org/abs/2003.03653
- 本地论文: `paper.pdf`
- 官方代码: https://github.com/TiagoCortinhal/SalsaNext
- 本地代码: `code/`
- 代码 commit: `7548c124b48f0259cdc40e98dfc3aeeadca6070c`
- 任务: LiDAR point cloud semantic segmentation
- 方法类型: range-image / projection-based encoder-decoder CNN
- 主要数据集: SemanticKITTI
- 核心模块: context module, residual dilated convolution stack, average pooling, pixel shuffle decoder, central dropout, weighted CE + Lovasz-Softmax, Bayesian uncertainty
- 复现备注: 官方代码基于 PyTorch/conda 环境，提供训练、评测、预训练权重和 uncertainty 版本；未下载 SemanticKITTI 或预训练权重。
