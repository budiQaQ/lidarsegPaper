# 3D-MiniNet 元信息

- 论文标题: 3D-MiniNet: Learning a 2D Representation from Point Clouds for Fast and Efficient 3D LiDAR Semantic Segmentation
- 作者: Inigo Alonso, Luis Riazuelo, Luis Montesano, Ana C. Murillo
- 会议/年份: IROS 2020
- arXiv: https://arxiv.org/abs/2002.10893
- 本地论文: `paper.pdf`
- 代码: https://github.com/Shathe/3D-MiniNet
- 本地代码: `code/`
- 本地 commit: `21a27838f3a43b3722406ed5a91fdef512f54910`
- 任务: LiDAR semantic segmentation
- 主要数据集: SemanticKITTI, KITTI
- 表示形式: learned 2D representation from 3D points + 2D FCNN
- 核心模块: projection learning module, local/global 3D features, MiniNet-style 2D FCNN, optional kNN post-processing
- 参数量与计算耗时: small 约 1.13M/61 FPS；标准版约 3.97M/36 FPS。
- 代表结果: SemanticKITTI test 3D-MiniNet-KNN 55.8 mIoU；small 版本 51.8 mIoU / 61 FPS / 1.13M params
- 复现备注: 仓库同时含 PyTorch 与 TensorFlow 代码；PyTorch 版本基于 RangeNet++，TensorFlow 版本基于 LU-Net。
