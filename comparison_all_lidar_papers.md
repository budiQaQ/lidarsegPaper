# LiDAR / Point Cloud 论文总对比

当前已整理论文：

- PointNet: raw point-based 基础方法
- PointNet++: raw point-based 层次化局部几何方法
- LU-Net: learned 3D local feature + range-image U-Net
- RangeNet++: spherical range image + Darknet + kNN post-processing
- SqueezeSegV3: spatially-adaptive convolution for range image
- 3D-MiniNet: learned 2D representation + light 2D FCNN
- SalsaNext: range-image encoder-decoder LiDAR segmentation
- FIDNet: range-image parameter-free interpolation decoding
- Lite-HDSeg: lite harmonic dense convolution + ICM + MCSPN
- KPRNet: projection CNN + KPConv point-wise refinement
- CENet: 基于 FIDNet 思路增强的 concise/efficient range-image LiDAR segmentation
- 2DPASS: training-only 2D priors assisted 3D LiDAR semantic segmentation

## 方法谱系

| 方法 | 表示形式 | 核心思想 | 主要任务 | 与其他方法的关系 |
| --- | --- | --- | --- | --- |
| PointNet | raw point set | shared MLP + max pooling 保证 permutation invariance | 分类、part segmentation、室内语义分割 | 点云深度学习基础方法；PointNet++ 直接继承并补齐局部结构 |
| PointNet++ | raw point set + local neighborhoods | FPS + ball query + local PointNet + feature propagation | 分类、part segmentation、scene parsing | raw point 层次化局部几何代表 |
| LU-Net | 3D local feature + spherical range image | 8 邻居 learned local feature，再输入 U-Net | KITTI LiDAR 三类语义分割 | 介于 raw point 与 range-image CNN 之间 |
| RangeNet++ | spherical range image | Darknet backbone + range-aware kNN post-processing | SemanticKITTI | 经典 projection-based baseline，后续大量代码/后处理继承对象 |
| SqueezeSegV3 | spherical range image | Spatially-Adaptive Convolution | SemanticKITTI | 在 RangeNet++ 框架上改卷积算子 |
| 3D-MiniNet | learned 2D representation | 从 3D 点学习 2D tensor，再用轻量 FCNN | SemanticKITTI / KITTI | 与 LU-Net 同属“先注入 3D 信息再 2D 分割”的混合路线 |
| SalsaNext | spherical range image | residual dilated encoder-decoder + pixel shuffle + uncertainty | SemanticKITTI | projection-based 代表，decoder 比 FIDNet/CENet 更完整 |
| FIDNet | spherical range image | bilinear interpolation 解码 + NLA 后处理 | SemanticKITTI | 极简 range-image baseline；CENet 明确 heavily based on FIDNet |
| Lite-HDSeg | spherical range image | harmonic dense conv + ICM + MCSPN + boundary loss | SemanticKITTI | 高精度实时 encoder-decoder，强调上下文和边界 |
| KPRNet | range image + raw points | 2D CNN 特征 + KPConv point-wise refinement | SemanticKITTI | 用可学习 3D refinement 替代固定 kNN/NLA |
| CENet | spherical range image | 3x3 conv + stronger activation + auxiliary heads + multi-loss | SemanticKITTI / SemanticPOSS | 在 FIDNet 基础上增强表达能力和训练监督 |
| 2DPASS | sparse point-voxel + training-only image branch | 2D priors + multi-scale fusion-to-single KD | SemanticKITTI / NuScenes | 不改变推理输入，只在训练期用图像先验增强 3D LiDAR 模型 |

## 关键技术维度对比

| 维度 | PointNet | PointNet++ | LU-Net | RangeNet++ | SqueezeSegV3 | 3D-MiniNet | SalsaNext | FIDNet | Lite-HDSeg | KPRNet | CENet | 2DPASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 表示 | raw point | raw point | 3D local + 2D | range image | range image | learned 2D | range image | range image | range image | 2D + 3D point | range image | sparse point-voxel + image teacher |
| 局部 3D 几何 | 弱 | 强 | 中强 | 弱 | 弱 | 中强 | 弱 | 弱 | 弱 | 强 | 弱 | 强 |
| 后处理/回填 | 无 | feature propagation | U-Net 输出回点 | kNN | kNN 可选 | kNN 可选 | kNN/uncertainty 相关流程 | NLA | MCSPN + kNN | KPConv refinement | NLA/KNN 风格 | 推理期纯 3D 输出 |
| 主要优势 | 理论基础 | 层次几何 | 早期实时混合 | 工程基线 | LiDAR 自适应卷积 | 轻量 learned projection | 完整上下文 + uncertainty | 极简 decoder | 高 mIoU/实时折中 | 点级可学习恢复 | 高精度/简洁折中 | 图像先验训练增强，推理不依赖图像 |
| 主要瓶颈 | 大场景弱 | 采样/邻域慢 | 老 TF 代码 | 精度偏低 | SAC 成本 | 分组复杂 | 网络较复杂 | 表达上限 | 无官方代码 | 速度慢 | 复现细节多 | 训练数据/依赖复杂 |

## 这次新增五篇与过往论文的主要不同

### RangeNet++ 带来的基线意义

RangeNet++ 是后续很多 range-image 方法的共同参照。它把流程拆成 spherical projection、2D CNN、range-aware kNN post-processing 三段，SqueezeSegV3 复用它的框架，3D-MiniNet 的 PyTorch 版基于它改造，FIDNet/CENet 也都绕不开它的 kNN 后处理设定。

实际差异：

- 相比 FIDNet/CENet，RangeNet++ 的 decoder/post-processing 更传统，精度较低但流程最清楚。
- 相比 KPRNet，RangeNet++ 用固定 kNN，KPRNet 用可学习 KPConv，后者精度更高但速度更慢。
- 相比 SqueezeSegV3，二者框架接近，主要差异来自标准卷积 vs SAC。

### SqueezeSegV3 的不同点及影响

SqueezeSegV3 的核心不是后处理，而是卷积算子。它认为 LiDAR range image 不同位置的特征分布差异很大，因此标准卷积共享权重不够合理，SAC 让卷积核随空间位置/输入自适应。

实际差异：

- 相比 RangeNet++，在相同框架下提升 mIoU，但增加额外 MAC 和参数。
- 相比 CENet，SqueezeSegV3 更偏算子创新；CENet 更偏普通卷积结构、激活函数和监督策略的工程组合。
- 对垂直方向分布差异明显的类别和区域，SAC 理论上更有优势。

### 3D-MiniNet 的不同点及影响

3D-MiniNet 不满足于固定 range projection，而是先从 3D 点学习 2D 表示，再交给轻量 FCNN。它和 LU-Net 都是“3D 信息前置注入”的路线。

实际差异：

- 相比 RangeNet++/FIDNet/CENet，3D-MiniNet 前端更复杂，但能减少固定投影带来的信息损失。
- 相比 LU-Net，3D-MiniNet 在 SemanticKITTI 上更系统验证，并提供 PyTorch/TF 两套实现。
- 相比 KPRNet，3D-MiniNet 在前端学习表示，KPRNet 在后端学习点级 refinement；二者都在弥补纯 2D projection 的几何不足。

### Lite-HDSeg 的不同点及影响

Lite-HDSeg 走高精度实时 encoder-decoder 路线，通过 harmonic dense convolution、ICM、MCSPN 和 boundary loss 同时增强特征、上下文和边界。

实际差异：

- 相比 SalsaNext，Lite-HDSeg 更强调 harmonic dense 和边界传播，mIoU 更高。
- 相比 CENet，Lite-HDSeg 模块更多，精度强但复现变量更多，且未找到可信官方代码。
- 相比 KPRNet，Lite-HDSeg 仍主要保持 2D range-image pipeline，更适合实时部署。

### KPRNet 的不同点及影响

KPRNet 把最后的回点过程变成可学习 KPConv point-wise refinement，这是对固定 kNN/NLA 的直接替代。

实际差异：

- 精度上 KPRNet 达到 63.1 mIoU，明显强于 RangeNet++/SqueezeSegV3/SalsaNext。
- 速度上 KPRNet 很难保持纯 range-image 方法的实时优势，适合高精度研究或离线/低频场景。
- 工程上需要处理点级邻域、KPConv 算子和 batch padding，复杂度高于 CENet/FIDNet。

### 2DPASS 的不同点及影响

2DPASS 与前面 range-image 论文不是同一类结构。它的主干是 sparse point-voxel 3D 网络，2D 图像只在训练期作为辅助分支，通过 multi-scale fusion-to-single knowledge distillation 把图像语义和边界先验迁移给 3D 网络。推理阶段不需要图像输入。

实际差异：

- 相比 CENet/FIDNet/RangeNet++/SqueezeSegV3/SalsaNext，2DPASS 精度更高，但代价是训练链路、数据准备和依赖环境明显更复杂。
- 相比 KPRNet，2DPASS 不在推理期增加点级 refinement；KPRNet 的 KPConv 每次推理都要运行，2DPASS 的 2D 分支训练后可丢弃。
- 相比 PointNet/PointNet++，2DPASS 面向完整自动驾驶大场景，主干更接近现代 sparse point-voxel LiDAR segmentation。
- 它对小目标和边界类提升明显，说明图像先验对 person、bicyclist、motorcyclist、pole、traffic-sign 等类别更有价值。

## 优劣结论

- 如果研究 raw point 基础理论，PointNet/PointNet++ 仍是必读。
- 如果研究 range-image 工程 baseline，RangeNet++ 是最重要起点。
- 如果研究轻量高效 decoder，FIDNet 和 CENet 更直接。
- 如果研究卷积算子适配 LiDAR range image，SqueezeSegV3 是关键参考。
- 如果研究 learned projection / 3D 信息前置，LU-Net 和 3D-MiniNet 应放在一起看。
- 如果研究高精度实时 encoder-decoder，Lite-HDSeg 和 SalsaNext 应放在一起看。
- 如果研究点级边界恢复和后处理替代，KPRNet 是最重要参考。
- 如果研究“训练期借用图像、推理期只用 LiDAR”的工程路线，2DPASS 是当前已整理论文中最重要参考。
- 当前工程 baseline 建议优先 CENet/FIDNet/RangeNet++；改进起点可按目标选择 CENet、Lite-HDSeg 或 KPRNet。
- 若允许复杂 3D sparse backbone 和图像辅助训练，2DPASS 的精度上限明显高于已整理的 range-image 组。

## 后续对比实验建议

- 统一数据集和 split，优先 SemanticKITTI single-scan。
- 统一是否启用 kNN/NLA/KPConv/MCSPN 等后处理，否则 mIoU 和速度不可横比。
- 对 projection-based 方法分段记录投影、网络、后处理、回写点云四段耗时。
- 类别级 IoU 重点看 person、bicyclist、motorcyclist、pole、traffic-sign 等小目标/细结构类别。
- KPRNet、3D-MiniNet、LU-Net 这类混合 2D/3D 方法要额外记录邻域构建和点分组成本。
- 2DPASS 需要额外记录图像数据准备、点到图像投影、2D branch 训练成本、TTA/instance augmentation 是否启用。
