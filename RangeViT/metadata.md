# RangeViT 元信息

- 论文标题: RangeViT: Towards Vision Transformers for 3D Semantic Segmentation in Autonomous Driving
- 作者: Angelika Ando, Spyros Gidaris, Andrei Bursuc, Gilles Puy, Alexandre Boulch, Renaud Marlet
- 会议/年份: CVPR 2023
- arXiv: https://arxiv.org/abs/2301.10222
- 本地论文: `paper.pdf`
- 代码: https://github.com/valeoai/rangevit
- 本地代码: `code/`
- 本地 commit: `4484d35b656f8213d887754780a4aa1e23303ea3`
- 任务: LiDAR semantic segmentation
- 主要数据集: nuScenes, SemanticKITTI
- 表示形式: range image + ViT encoder + convolutional stem/decoder + 3D refiner
- 核心模块: image-pretrained ViT transfer, small rectangular patch tokenization, UpConv decoder, 3D refiner
- 代表结果:
  - SemanticKITTI test: 64.0 mIoU
  - nuScenes validation: 75.2 mIoU with Cityscapes-pretrained ViT
- 复现备注: README 提供预训练模型入口，本次未下载模型权重。
