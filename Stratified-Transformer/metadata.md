# Stratified Transformer 元信息

- 论文标题: Stratified Transformer for 3D Point Cloud Segmentation
- 作者: Xin Lai, Jianhui Liu, Li Jiang, Liwei Wang, Hengshuang Zhao, Shu Liu, Xiaojuan Qi, Jiaya Jia
- 会议/年份: CVPR 2022
- arXiv: https://arxiv.org/abs/2203.14508
- 本地论文: `paper.pdf`
- 代码: https://github.com/dvlab-research/Stratified-Transformer
- 本地代码: `code/`
- 本地 commit: `140783d3d8e93a27a8724c69fd952fac0645be4d`
- 任务: 3D point cloud semantic segmentation
- 主要数据集: S3DIS, ScanNetv2
- 表示形式: point-based Transformer
- 核心模块: stratified key sampling, dense nearby keys + sparse distant keys, memory-efficient variable-length token implementation
- 代表结果: S3DIS 和 ScanNetv2 上达到当时 SOTA；README 强调 point-based 方法首次超过 voxel-based 方法
- 复现备注: 代码依赖自定义 CUDA pointops2，主要面向室内数据集；不是自动驾驶 LiDAR range-view 方法。
