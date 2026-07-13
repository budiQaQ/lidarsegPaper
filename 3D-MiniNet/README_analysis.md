# 3D-MiniNet 解读

## 一句话总结

3D-MiniNet 是介于 raw point 与 range-image CNN 之间的混合路线：先从 3D 点学习一个 2D 表示，再交给轻量 2D FCNN 做语义分割。

## 方法动机

纯 range-image 方法速度快但投影损失明显；纯 3D 点云方法几何表达强但推理较慢。3D-MiniNet 试图把两者折中：用 projection learning module 把局部/全局 3D 信息编码进 2D tensor，再利用 2D CNN 的高效推理。

## 网络结构与关键模块

- projection learning module 从 raw points 学习 2D representation。
- 该模块包含局部上下文、相对特征、attention MLP 等组成部分。
- 后端使用 MiniNet-style 轻量 2D FCNN。
- 代码中 PyTorch 版基于 RangeNet++ 目录结构，`backbones/3dmininet.py` 中可见点分组、1x1 conv、attention 和深度可分离卷积结构。

## 输入/输出与投影表示

- 输入是完整 LiDAR scan，先组织成点组并学习 2D 表示。
- 训练/推理中仍保留 range image 的投影索引，用于输出回填到点。
- 输出为点级语义标签。

## Loss / post-processing / training strategy

- 使用 cross entropy loss。
- 使用 SGD，论文报告训练 500 epochs。
- 可选 kNN post-processing，论文中 3D-MiniNet-KNN 是最终较强结果。

## 实验数据集与指标

- 3D-MiniNet-small: 51.8 mIoU，61 FPS，1.13M 参数。
- 3D-MiniNet: 53.0 mIoU，36 FPS，3.97M 参数。
- 3D-MiniNet-KNN: 55.8 mIoU。
- 相比 RangeNet53++ 52.2，KNN 版本提升约 3.6 mIoU，同时保持较高速度。

## 优点

- 兼顾 3D 几何和 2D CNN 效率，谱系上非常适合连接 LU-Net 与 RangeNet++。
- 参数量小，速度快，适合作轻量实时 baseline。
- 代码同时给出 PyTorch 和 TensorFlow 两套实现，方便比较 RangeNet++ 与 LU-Net 框架。

## 缺点

- 结构比纯 range-image baseline 更复杂，涉及点分组和 learned projection。
- 最终强结果仍依赖 kNN post-processing。
- 论文指标低于后续 SalsaNext、FIDNet、CENet、Lite-HDSeg、KPRNet。

## 工程复现风险

- 点分组、batch 形状和投影索引处理容易出错。
- PyTorch 版继承 RangeNet++ 老框架，TensorFlow 版继承 LU-Net 老框架，依赖维护成本较高。
- 速度报告需确认是否包含投影、点分组和 kNN 后处理。

## 适合借鉴的实现点

- 用 learned projection 替代固定 spherical projection 的部分信息瓶颈。
- 对轻量模型做参数量、FPS、mIoU 三者联合比较。
- 将局部 3D 信息注入 2D tensor，再使用常规 2D FCNN。

## 与后续论文对比时应关注的维度

- 与 LU-Net: 都是先补 3D 信息再走 2D CNN；3D-MiniNet 更系统地学习 2D representation。
- 与 RangeNet++: learned projection + MiniNet FCNN vs fixed projection + Darknet。
- 与 FIDNet/CENet: 输入表示增强 vs decoder/backbone/loss 增强。
- 与 KPRNet: 前端 learned 2D representation vs 后端 KPConv point-wise refinement。

## 论文与代码地址

- 论文地址: https://arxiv.org/abs/2002.10893
- GitHub 仓库: https://github.com/Shathe/3D-MiniNet

## 核心创新代码块

```python
# 3D-MiniNet/code/pytorch_code/lidar-bonnetal/train/tasks/semantic/modules/trainer.py
outputs = model([in_vol, proj_chan_group_points], proj_mask)
loss = criterion(torch.log(outputs.clamp(min=1e-8)), proj_labels)
```

## 使用方法描述

使用时同时输入 range image 特征 `in_vol` 和分组后的点级局部特征 `proj_chan_group_points`，先学习 2D 表示，再由轻量 2D FCNN 输出语义。

