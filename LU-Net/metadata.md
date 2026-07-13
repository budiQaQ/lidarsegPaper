# LU-Net 元信息

- 论文标题: LU-Net: An Efficient Network for 3D LiDAR Point Cloud Semantic Segmentation Based on End-to-End-Learned 3D Features and U-Net
- 作者: Pierre Biasutti, Vincent Lepetit, Mathieu Bredif, Jean-Francois Aujol, Aurelie Bugeau
- 年份: arXiv 2019
- arXiv: https://arxiv.org/abs/1908.11656
- 本地论文: `paper.pdf`
- 代码来源: https://github.com/budiQaQ/lunet
- 原始 README 指向: https://github.com/pbias/lunet
- 本地代码: `code/`
- 代码 commit: `7c802f19fbf53f5124d4d4785c6b0355d8771f6d`
- 任务: LiDAR point cloud semantic segmentation
- 方法类型: hybrid 3D local feature extraction + range-image U-Net
- 主要数据集: KITTI 3D object detection dataset 上的 LiDAR 语义分割设置
- 核心模块: sensor topology, 8-connected 3D neighborhood, learned local 3D feature extraction, multichannel range image, U-Net segmentation network, focal loss
- 参数量与计算耗时: LU-Net 论文报告约 24 FPS；参数量需在 TF1 环境中运行官方代码统计。
- 复现备注: 代码基于 Python 3.6、TensorFlow 1.6 GPU；需要先下载 SqueezeSeg 数据集，并在 `make_tfrecord_pn.py` 中手动设置 `semantic_base` 路径，再生成 TFRecords。默认配置为 `64 x 512` range image、`n_size=[3,3]`、`channels=xyzdr`、`pointnet=True`、`n_classes=4`、`focal_loss=True`。
