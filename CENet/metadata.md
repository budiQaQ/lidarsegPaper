# CENet 元信息

- 论文标题: CENet: Toward Concise and Efficient LiDAR Semantic Segmentation for Autonomous Driving
- 作者: Huixian Cheng, Xianfeng Han, Guoqiang Xiao
- 会议/年份: IEEE ICME 2022
- arXiv: https://arxiv.org/abs/2207.12691
- 本地论文: `paper.pdf`
- 官方代码: https://github.com/huixiancheng/CENet
- 本地代码: `code/`
- 代码 commit: `9a84103d186a1f93637cae3d96426760deb04140`
- 任务: LiDAR point cloud semantic segmentation
- 方法类型: range-image / spherical projection based 2D CNN
- 主要数据集: SemanticKITTI, SemanticPOSS
- 核心模块: 3x3 convolution input/classification head, ResNet-style backbone, FID-like multi-scale interpolation fusion, Hardswish/SiLU activation, auxiliary segmentation heads, weighted CE + Lovasz-Softmax + boundary loss
- 复现备注: README 提供训练、推理、评测命令和 Google Drive 预训练权重；未下载数据集和权重。作者 README 说明部分模型和 config 可能因多次更新存在不一致，需要按具体报错调整。
