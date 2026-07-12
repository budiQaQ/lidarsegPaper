# Point Transformer V3 元信息

- 论文标题: Point Transformer V3: Simpler, Faster, Stronger
- 作者: Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao
- 会议/年份: CVPR 2024
- arXiv: https://arxiv.org/abs/2312.10035
- 本地论文: `paper.pdf`
- 代码: https://github.com/Pointcept/PointTransformerV3
- 本地代码: `code/`
- 本地 commit: `3229e9b7de1770c8ad17c316f8e349982de509f8`
- 任务: 3D point cloud perception backbone；semantic segmentation, instance segmentation, detection
- 主要数据集: ScanNet, ScanNet200, S3DIS, nuScenes, Waymo, SemanticKITTI
- 表示形式: serialized point cloud Transformer
- 核心模块: point cloud serialization, wider receptive field, FlashAttention, simplified point Transformer
- 代表结果: nuScenes validation 80.3 mIoU；Waymo semantic segmentation 71.2 mIoU；论文图中 SemanticKITTI semantic segmentation 63.5
- 复现备注: README 指向 Pointcept v1.5 作为主框架；本地 repo 可单独复制 `model.py` 和 `serialization/`。
