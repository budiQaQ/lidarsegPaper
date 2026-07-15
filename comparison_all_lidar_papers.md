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
- RangeViT: range-image ViT for LiDAR semantic segmentation
- RangeFormer: high-performance LiDAR-only range-view Transformer framework
- RangeRet: lightweight range-view Retentive Network with Circular Retention
- FLARES: fast and accurate multi-range range-view training/inference paradigm
- Swin Transformer: 通用 shifted-window vision Transformer，可作为 camera teacher / range window attention 参考
- Point Transformer V3: 现代 3D point Transformer 强基线
- Stratified Transformer: point-based long-range context Transformer

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
| RangeViT | range image | image-pretrained ViT + stem/decoder/refiner | SemanticKITTI / nuScenes | 证明 ViT/RGB 预训练可迁移到 range-view LiDAR |
| RangeFormer | range image | RangeFormer + RangeAug + RangePost + STR | SemanticKITTI / nuScenes / ScribbleKITTI | 当前最关键的 range-view Transformer 参考 |
| RangeRet | range image | Retentive Network + Circular Retention | SemanticKITTI / PandaSet / SemanticPOSS | 更轻量的 range-view 长程建模路线，适合作 LC 蒸馏 student |
| FLARES | multi-range image | spherical-coordinate splitting + low-resolution multi-range projection + WPD+/MCF + NNRI | SemanticKITTI / nuScenes | 不替换 backbone，而是增强 SalsaNext/FIDNet/CENet/RangeViT 的 range-view pipeline |
| Swin Transformer | RGB image | shifted window hierarchical Transformer | 通用视觉任务 | 不是 LiDAR 方法；适合做 camera teacher 或 range window attention 模块 |
| Point Transformer V3 | serialized 3D points | 更简单高效的 point Transformer | 多个 3D perception benchmark | 3D Transformer 强对照，可作 3D teacher |
| Stratified Transformer | 3D points | 近处密集 + 远处稀疏 key sampling | S3DIS / ScanNetv2 | 长程上下文采样思想可迁移到 range-view |

## 关键技术维度对比

| 维度 | PointNet | PointNet++ | LU-Net | RangeNet++ | SqueezeSegV3 | 3D-MiniNet | SalsaNext | FIDNet | Lite-HDSeg | KPRNet | CENet | 2DPASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 表示 | raw point | raw point | 3D local + 2D | range image | range image | learned 2D | range image | range image | range image | 2D + 3D point | range image | sparse point-voxel + image teacher |
| 局部 3D 几何 | 弱 | 强 | 中强 | 弱 | 弱 | 中强 | 弱 | 弱 | 弱 | 强 | 弱 | 强 |
| 后处理/回填 | 无 | feature propagation | U-Net 输出回点 | kNN | kNN 可选 | kNN 可选 | kNN/uncertainty 相关流程 | NLA | MCSPN + kNN | KPConv refinement | NLA/KNN 风格 | 推理期纯 3D 输出 |
| 主要优势 | 理论基础 | 层次几何 | 早期实时混合 | 工程基线 | LiDAR 自适应卷积 | 轻量 learned projection | 完整上下文 + uncertainty | 极简 decoder | 高 mIoU/实时折中 | 点级可学习恢复 | 高精度/简洁折中 | 图像先验训练增强，推理不依赖图像 |
| 主要瓶颈 | 大场景弱 | 采样/邻域慢 | 老 TF 代码 | 精度偏低 | SAC 成本 | 分组复杂 | 网络较复杂 | 表达上限 | 无官方代码 | 速度慢 | 复现细节多 | 训练数据/依赖复杂 |

## 参数量与计算耗时对比

注意：下表只汇总论文/官方 README/已整理材料中可确认的口径。不同论文硬件、输入分辨率、是否包含 projection/post-processing、是否使用 FP16/TTA 都不同，因此只能用于工程筛选，不能当作严格 benchmark。

| 方法 | 参数量 | 计算耗时 / 速度 | 口径备注 |
| --- | --- | --- | --- |
| PointNet | 未给出统一 LiDAR segmentation 参数量 | 未报告整帧 LiDAR FPS | 原始任务是分类、part segmentation、室内场景，不是自动驾驶 scan 级 benchmark |
| PointNet++ | 未给出统一参数量 | 未报告可横比整帧 LiDAR FPS | SSG/MSG/MRG、点数、采样邻域会显著改变耗时 |
| LU-Net | 未报告；代码可在 TF1 环境打印 trainable variables | 约 24 FPS，约 41.7 ms/scan | 早期 KITTI 三类设置，不能直接横比 SemanticKITTI 20 类 |
| RangeNet++ | 本地官方 README 未确认 | RangeNet53++ 约 12 scans/sec，约 83 ms/scan | 需确认是否包含 kNN post-processing |
| SqueezeSegV3 | 未确认；随 SAC/ResNet-21/53 变化 | SqueezeSegV3-53+kNN 约 6 scans/sec，约 167 ms/scan | SAC 增加 MAC 和显存，kNN 开关影响速度 |
| 3D-MiniNet | small 1.13M；标准版 3.97M | small 61 FPS，标准版 36 FPS | KNN 版本精度更高，但后处理耗时需单独统计 |
| SalsaNext | 后续对比表约 6.7M | 后续对比表约 19 ms | PandaSet validation 对比口径，非所有论文同硬件 |
| FIDNet | 约 6M | CNN-only 约 11 ms；NLA 约 1.2 ms；FP16 单帧约 0.01 s | decoder 无参数，后处理 NLA 比 KNN 更轻 |
| Lite-HDSeg | 未确认 | 约 20 FPS，约 50 ms/scan | 无可信官方代码，本地不估算参数 |
| KPRNet | 未确认 | 对比表约 0.3 FPS，约 3333 ms/scan | KPConv point-wise refinement 是主要耗时来源 |
| CENet | 代码可统计，当前未运行 PyTorch | 64x2048 约 37.79 FPS；64x1024 约 67.97 FPS；64x512 约 84.91 FPS | 官方 RTX 3080 speed log，含 CNN 与轻量 KNN/NLA 风格后处理 |
| 2DPASS | 1.9M 小模型；45.6M 大模型 | SemanticKITTI single-scan 约 62 ms | 推理期只保留 3D LiDAR student，图像分支只在训练期 |
| RangeViT | 未确认；随 ViT backbone/patch/decoder 变化 | 未找到可横比单帧 latency | 需要本地拆分 stem、ViT、decoder/refiner 统计 |
| RangeFormer | 约 23.7M | 约 54 ms | RangeRet 论文 PandaSet validation 对比表口径 |
| RangeRet | 约 3.8M | 约 38 ms | 轻量 Retentive Network，显式 circular prior |
| FLARES | 取决于 backbone；论文表中 FLARES-FIDNet 约 6.0M，FLARES-CENet 约 6.8M，FLARES-RangeViT 约 24.1M | FLARES-FIDNet 约 19 ms，FLARES-CENet 约 22 ms，FLARES-SalsaNext/RangeViT 约 29 ms | 多范围 pipeline，论文报告超过 40% inference speed-up |
| Swin Transformer | Swin-T/S/B backbone 约 28M/50M/88M；UPerNet 约 60M/81M/121M | ImageNet 分类 Swin-T/S/B 约 755/437/278 FPS | 通用视觉模型，不能直接代表 camera teacher segmentation latency |
| Point Transformer V3 | 未确认；随 Pointcept 配置变化 | 论文整理中约 44 ms，显存约 1.2G | 3D point Transformer 口径，不是 range-view student |
| Stratified Transformer | 未确认 | 未找到可横比 LiDAR scan latency | 室内 point-based benchmark，依赖 pointops2 |

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

### FLARES 的不同点及影响

FLARES 不是新 backbone，而是 range-view pipeline 的重设计。它把一帧点云按 spherical coordinate 拆成多个 sub-cloud，再投影成多张低分辨率 range image；训练阶段加入 WPD+ 和 MCF，推理阶段用 NNRI 做 multi-range 回投影。

实际差异：

- 相比 CENet/FIDNet，FLARES 不主要改网络结构，而是改输入表示、增强和后处理；因此可以直接作为 CENet/FIDNet 的外层 pipeline 改造方向。
- 相比 RangeFormer/STR，FLARES 更强调低分辨率 multi-range projection 和专门的 NNRI 后处理；STR 更像训练尺度压缩策略，FLARES 同时追求精度和推理速度。
- 相比 RangeNet++/FIDNet 的 KNN/NLA，NNRI 使用 softmax score、range difference 和距离自适应阈值做插值，更适合低分辨率多 range image。
- 相比 2DPASS，FLARES 不依赖相机图像；但 WPD+ 可引入 Carla synthetic data 来缓解长尾类，这会带来额外数据和 domain gap 变量。
- 工程上它最适合和 CENet/FIDNet 联合验证：先不动 backbone，只替换 projection/splitting/post-processing，即可观察 multi-range 范式收益。

## 优劣结论

- 如果研究 raw point 基础理论，PointNet/PointNet++ 仍是必读。
- 如果研究 range-image 工程 baseline，RangeNet++ 是最重要起点。
- 如果研究轻量高效 decoder，FIDNet 和 CENet 更直接。
- 如果研究卷积算子适配 LiDAR range image，SqueezeSegV3 是关键参考。
- 如果研究 learned projection / 3D 信息前置，LU-Net 和 3D-MiniNet 应放在一起看。
- 如果研究高精度实时 encoder-decoder，Lite-HDSeg 和 SalsaNext 应放在一起看。
- 如果研究点级边界恢复和后处理替代，KPRNet 是最重要参考。
- 如果研究“训练期借用图像、推理期只用 LiDAR”的工程路线，2DPASS 是当前已整理论文中最重要参考。
- 如果研究 range-view pipeline 级别的精度/速度折中，FLARES 是当前最值得补进 CENet/FIDNet baseline 的新方向。
- 当前工程 baseline 建议优先 CENet/FIDNet/RangeNet++；改进起点可按目标选择 CENet、Lite-HDSeg 或 KPRNet。
- 若允许复杂 3D sparse backbone 和图像辅助训练，2DPASS 的精度上限明显高于已整理的 range-image 组。
- 如果研究 range-view Transformer，优先看 RangeFormer 和 RangeViT。
- 如果研究轻量 range-view 长程建模或部署友好 student，RangeRet 值得和 CENet/FIDNet 放在一起比较。
- 如果研究 LC 蒸馏中的 camera teacher 或窗口 attention，Swin 是结构参考。
- 如果需要 3D Transformer teacher/upper bound，对比 PTv3；如果关注低成本长程上下文，参考 Stratified Transformer。

## 后续对比实验建议

- 统一数据集和 split，优先 SemanticKITTI single-scan。
- 统一是否启用 kNN/NLA/KPConv/MCSPN 等后处理，否则 mIoU 和速度不可横比。
- 对 projection-based 方法分段记录投影、网络、后处理、回写点云四段耗时。
- 类别级 IoU 重点看 person、bicyclist、motorcyclist、pole、traffic-sign 等小目标/细结构类别。
- KPRNet、3D-MiniNet、LU-Net 这类混合 2D/3D 方法要额外记录邻域构建和点分组成本。
- 2DPASS 需要额外记录图像数据准备、点到图像投影、2D branch 训练成本、TTA/instance augmentation 是否启用。
- Transformer 方法要单独记录 token 数、window/patch shape、attention 类型、预训练来源和是否依赖 FlashAttention/custom CUDA。
