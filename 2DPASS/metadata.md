# 2DPASS 元信息

- 论文标题: 2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds
- 作者: Xu Yan, Jiantao Gao, Chaoda Zheng, Chao Zheng, Ruimao Zhang, Shuguang Cui, Zhen Li
- 会议/年份: ECCV 2022
- arXiv: https://arxiv.org/abs/2207.04397
- 本地论文: `paper.pdf`
- 代码: https://github.com/yanx27/2DPASS
- 本地代码: `code/`
- 本地代码来源: GitHub main 分支 zip 源码包；`git clone` 两次因网络失败，改用 zip 下载
- GitHub main commit: `80b8646bbb0dd46ddcee31f531d6f1c45baf35fe`
- 任务: LiDAR semantic segmentation
- 主要数据集: SemanticKITTI, NuScenes
- 表示形式: 3D sparse point/voxel backbone + training-only 2D image branch
- 核心模块: auxiliary 2D branch, multi-scale fusion-to-single knowledge distillation, SPVCNN-style 3D baseline
- 代表结果:
  - SemanticKITTI single-scan test: 72.9 mIoU，论文表中速度约 62 ms
  - SemanticKITTI multiple-scan test: 62.4 mIoU
  - 代码 README model zoo: SemanticKITTI validation 2DPASS large 70.7 mIoU / TTA 72.0；NuScenes validation large 78.0 / TTA 80.5
- 复现备注: 训练需要 SemanticKITTI 点云与 KITTI odometry color images；推理阶段不需要图像输入。README 提供 Google Drive 权重入口，本次未下载权重和数据集。
