# RangeFormer 元信息

- 论文标题: Rethinking Range View Representation for LiDAR Segmentation
- 方法名: RangeFormer
- 作者: Lingdong Kong, Youquan Liu, Runnan Chen, Yuexin Ma, Xinge Zhu, Yikang Li, Yuenan Hou, Yu Qiao, Ziwei Liu
- 会议/年份: ICCV 2023
- arXiv: https://arxiv.org/abs/2303.05367
- 本地论文: `paper.pdf`
- 代码: 未发现可信官方公开代码；Papers with Code 当前也未提供官方代码入口
- 任务: LiDAR semantic segmentation / panoptic segmentation
- 主要数据集: SemanticKITTI, nuScenes, ScribbleKITTI
- 表示形式: range view / range image
- 核心模块: RangeFormer, RangeAug, RangePost, Scalable Training from Range view (STR)
- 参数量与计算耗时: 后续对比表口径约 23.7M/54 ms。
- 代表结果:
  - SemanticKITTI semantic segmentation test: 73.3 mIoU
  - SemanticKITTI panoptic segmentation: P-RangeFormer 64.2 PQ / 72.0 mIoU
  - STR 低分辨率训练: RangeFormer w/ STR 72.2 mIoU
- 复现备注: 无官方代码，短期更适合作方法设计参考；不能直接作为工程 baseline。
