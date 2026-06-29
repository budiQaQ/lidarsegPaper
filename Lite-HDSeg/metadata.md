# Lite-HDSeg 元信息

- 论文标题: Lite-HDSeg: LiDAR Semantic Segmentation Using Lite Harmonic Dense Convolutions
- 作者: Ryan Razani, Ran Cheng, Ehsan Taghavi, Liu Bingbing
- 单位: Huawei Noah's Ark Lab, Toronto
- 年份: 2021
- arXiv: https://arxiv.org/abs/2103.08852
- 本地论文: `paper.pdf`
- 代码: 未发现可信官方公开代码；本次未下载代码仓库。
- 任务: LiDAR semantic segmentation
- 主要数据集: SemanticKITTI
- 表示形式: spherical range image
- 核心模块: Lite Harmonic Dense Convolution, encoder-decoder, ICM, MCSPN, boundary loss, kNN post-processing
- 代表结果: SemanticKITTI test 63.8 mIoU, 约 20 FPS
- 复现备注: 由于未找到官方代码，工程复现需要按论文重实现，风险高于 RangeNet++/SqueezeSegV3/CENet/FIDNet。
