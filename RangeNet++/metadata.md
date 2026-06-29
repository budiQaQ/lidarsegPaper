# RangeNet++ 元信息

- 论文标题: RangeNet++: Fast and Accurate LiDAR Semantic Segmentation
- 作者: Andres Milioto, Ignacio Vizzo, Jens Behley, Cyrill Stachniss
- 会议/年份: IROS 2019
- 论文: http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf
- 本地论文: `paper.pdf`
- 代码: https://github.com/PRBonn/lidar-bonnetal
- 本地代码: `code/`
- 本地 commit: `99b827f0228ff0e997473ac8e2cecbaa4af7d7c7`
- 任务: LiDAR semantic segmentation
- 主要数据集: SemanticKITTI
- 表示形式: spherical range image
- 核心模块: Darknet-style backbone, spherical projection, range-aware kNN post-processing
- 代表结果: RangeNet53++ 64 x 2048 SemanticKITTI test 52.2 mIoU, 约 12 scans/sec
- 复现备注: 仓库已 archive；clone 到 macOS 大小写不敏感文件系统时出现 README/readme 名称冲突警告，但核心代码已下载。
