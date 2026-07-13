# FIDNet 元信息

- 论文标题: FIDNet: LiDAR Point Cloud Semantic Segmentation with Fully Interpolation Decoding
- 作者: Yiming Zhao, Lin Bai, Xinming Huang
- 会议/年份: IROS 2021 / arXiv 2021
- arXiv: https://arxiv.org/abs/2109.03787
- 本地论文: `paper.pdf`
- 官方代码: https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI
- 本地代码: `code/`
- 代码 commit: `7f90b45a765b8bba042b25f642cf12d8fccb5bc2`
- 任务: LiDAR point cloud semantic segmentation
- 方法类型: range-image / spherical projection based 2D CNN
- 主要数据集: SemanticKITTI
- 核心模块: 1x1 input module, ResNet-34 backbone, FID fully interpolation decoding, ASPP-like classification head, NLA nearest label assignment post-processing
- 参数量与计算耗时: 约 6M；CNN-only 约 11 ms，NLA 后处理约 1.2 ms，半精度单帧约 0.01 s。
- 复现备注: README 提供 Docker/PyTorch 1.7.1 环境、训练/验证/测试脚本和 Google Drive 预训练权重；未下载数据集和权重。
