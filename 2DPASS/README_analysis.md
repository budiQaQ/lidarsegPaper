# 2DPASS 解读

## 一句话总结

2DPASS 不是传统多模态融合推理方法，而是“训练期用 2D 图像先验增强 3D LiDAR 模型，推理期只用点云”的 cross-modal knowledge distillation 框架。

## 方法动机

相机图像有纹理、颜色和语义边界信息，LiDAR 点云有准确几何和距离信息。直接多模态融合通常要求训练和推理时都具备严格对齐的图像-点云输入，这在工程部署中不总是可靠。2DPASS 的核心思路是：训练时利用图像分支作为辅助老师，把 2D 语义/结构信息蒸馏给 3D 网络；推理时移除 2D 分支，只保留 3D LiDAR 网络。

## 网络结构与关键模块

- 3D 主干是 SPVCNN-style sparse point-voxel 网络，代码中 `network/baseline.py` 包含 voxelization、SPVBlock、point encoder 和 sparse convolution。
- 2D 分支是 ResNetFCN，在训练期处理裁剪后的相机图像 patch。
- `xModalKD` 是核心蒸馏模块，代码位于 `network/arch_2dpass.py`。
- 多尺度特征来自 `scale_list`，SemanticKITTI 默认为 `[2, 4, 8, 16]`。
- 每个尺度上先分别产生 3D prediction 和 2D-3D fusion prediction，再用 KL divergence 将 fusion prediction 的知识迁移给 3D prediction。

## 输入/输出与数据组织

- 训练输入包括 LiDAR 点云、语义标签、相机图像、点到图像的投影索引。
- SemanticKITTI 训练还需要 KITTI odometry color data，即 `image_2/`。
- 推理输入只需要 LiDAR 点云；代码中 `forward()` 只有在 `self.training and not self.baseline_only` 时才调用 2D branch 和 fusion。
- 输出是 3D 点级语义标签。

## Loss / post-processing / training strategy

- 3D 主任务 loss 使用 cross entropy + Lovasz。
- 2D/fusion 分支也有 segmentation loss。
- 跨模态蒸馏使用 `KLDiv(log_softmax(3D), softmax(fusion.detach()))`，即 fusion branch 作为教师信号，3D branch 作为学生。
- 配置中 `lambda_seg2d=1`，`lambda_xm=0.05`。
- 支持 `--baseline_only` 进行无 2D 分支的 vanilla 训练。
- 测试支持 TTA，README 提到 `num_vote=12`，`num_vote=1` 约有 2% 性能下降。

## 实验数据集与指标

- SemanticKITTI single-scan test: 72.9 mIoU。
- 论文中的 baseline 为 67.4 mIoU，2DPASS 带来约 +5.5 mIoU。
- SemanticKITTI multiple-scan test: 62.4 mIoU。
- NuScenes 也报告 SOTA 级结果；README model zoo 中 NuScenes validation large 为 78.0 mIoU，TTA 为 80.5。
- 代码 README 更新过更强模型，SemanticKITTI validation large 为 70.7 mIoU，TTA 为 72.0。

## 优点

- 推理阶段只用 LiDAR，不依赖相机在线可用性和严格同步，对工程部署更友好。
- 2D 图像先验明显改善小目标和边界类别，论文 qualitative 也强调 small objects 和 region boundaries。
- 框架具备通用性，原则上可接到不同 3D segmentation backbone。
- 代码提供 SemanticKITTI、NuScenes、MinkowskiNet、SPVCNN、robustness evaluation 等配置，工程材料较完整。

## 缺点

- 训练数据准备更复杂，需要点云、标签、图像和标定/投影关系。
- 训练期显存和时间成本高于纯 LiDAR 模型，因为需要 2D branch 和多尺度 fusion/KD。
- 它不是 range-image 方法，不能直接作为 CENet/FIDNet/RangeNet++ 的结构替代；更多是训练策略和 3D backbone 增强。
- 性能表中使用额外验证集、TTA、instance augmentation 等设置时，需要和普通论文结果分开比较。

## 工程复现风险

- `git clone` 在本机网络环境下失败，已用 GitHub zip 源码替代；无本地 `.git`，commit 来自 GitHub API。
- 依赖包含 PyTorch Lightning、torch-scatter、spconv、torchsparse，环境搭建成本高。
- SemanticKITTI 训练除点云外还要下载 KITTI odometry color images；本次按要求未下载数据集。
- README 中 NuScenes 训练命令配置文件名疑似有拼写差异，实际仓库文件是 `config/2DPASS-nuscenes.yaml`。

## 适合借鉴的实现点

- 对已有 3D LiDAR backbone 增加 training-only 2D teacher，而不增加推理时传感器依赖。
- 多尺度 fusion-to-single KD：每个尺度都做 3D head、fusion head 和 KL 蒸馏。
- 用 `--baseline_only` 保留同代码同配置下的 ablation baseline。
- 对小目标/边界类别做专门类别级 IoU 对比，验证图像先验是否真正带来结构收益。

## 与过往论文对比时应关注的维度

- 与 CENet/FIDNet/RangeNet++/SqueezeSegV3/SalsaNext: 2DPASS 不是 range-image 2D CNN，而是 sparse point-voxel 3D 网络 + 训练期图像蒸馏；精度更高，但训练复杂度高很多。
- 与 KPRNet: KPRNet 用 3D KPConv 在推理期做点级 refinement；2DPASS 用 2D 图像在训练期做知识迁移，推理期没有图像/KD 分支。
- 与 3D-MiniNet/LU-Net: 都试图弥补纯 2D/range 投影的信息损失；2DPASS 更偏 3D backbone 与跨模态训练，不是 learned range representation。
- 与 PointNet/PointNet++: 2DPASS 面向完整自动驾驶 LiDAR 大场景，主干使用 sparse point-voxel 结构，远强于早期 raw point set baseline。

## 论文与代码地址

- 论文地址: https://arxiv.org/abs/2207.04397
- GitHub 仓库: https://github.com/yanx27/2DPASS

## 核心创新代码块

```python
# 2DPASS/code/network/arch_2dpass.py
class get_model(LightningBaseModel):
    def forward(self, data_dict):
        data_dict = self.model_3d(data_dict)
        if self.training and not self.baseline_only:
            data_dict = self.model_2d(data_dict)
            data_dict = self.fusion(data_dict)
        return data_dict
```

## 使用方法描述

训练时不要用 `--baseline_only`，让 2D image branch 和 `xModalKD` 参与训练；推理/测试时只走 `model_3d`，因此部署仍是 LiDAR-only。

