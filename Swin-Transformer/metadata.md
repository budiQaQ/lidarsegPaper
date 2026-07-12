# Swin Transformer 元信息

- 论文标题: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
- 作者: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
- 会议/年份: ICCV 2021
- arXiv: https://arxiv.org/abs/2103.14030
- 本地论文: `paper.pdf`
- 代码: https://github.com/microsoft/Swin-Transformer
- 本地代码: `code/`
- 本地 commit: `f82860bfb5225915aca09c3227159ee9e1df874d`
- 任务: 通用视觉 backbone；分类、检测、实例分割、语义分割
- 核心模块: shifted window self-attention, hierarchical patch merging, relative position bias
- 代表结果: ADE20K semantic segmentation 53.5 mIoU；COCO test-dev 58.7 box AP / 51.1 mask AP
- 复现备注: 这不是 LiDAR segmentation 论文；本次作为 range-view LC 蒸馏中的 2D teacher / window attention 设计参考。
