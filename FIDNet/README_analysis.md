# FIDNet 解读

## 一句话总结

FIDNet 是一个极简 range-image LiDAR 语义分割框架，核心贡献是用无参数的 bilinear interpolation 做多尺度解码，并用 NLA 后处理替代更重的 KNN，从而在 SemanticKITTI 上取得较好的速度/精度平衡。

## 方法动机

很多 projection-based 方法沿用 RGB 图像分割里的 encoder-decoder 设计，但 LiDAR range image 的每个像素对应一个带几何含义的点。FIDNet 作者认为没必要使用复杂 decoder 或转置卷积，只要把不同层级的点特征插值回同一分辨率并拼接，就能获得足够的多尺度表达。

FIDNet 的另一个动机是重新审视后处理。传统 KNN 后处理同时处理 CNN 边界模糊和投影 many-to-one mapping，但 FIDNet 观察到自身网络输出边界已经较清晰，主要问题是多个点落在同一 range-image cell 后未被网络直接预测。因此提出 NLA，只给未直接预测的遮挡点分配最近点标签。

## 网络结构与关键模块

- 输入模块使用 1x1 convolution，将点的 `(x, y, z, range, remission)` 等特征映射到高维通道。
- backbone 使用 ResNet-34 风格结构提取多尺度特征。
- FID 模块将 stem 和各 stage 特征通过 bilinear interpolation 上采样到原始 range image 分辨率，然后 concatenate。
- classification head 采用 ASPP-like 设计，用 dilation rate 3/6/9 的 3x3 atrous conv 聚合局部上下文，再用 1x1 conv 输出类别。
- post-processing 默认可选 `if_KNN=2`，即作者提出的 NLA；`if_KNN=1` 是传统 KNN，`if_KNN=0` 是无后处理。

## 输入/输出与投影表示

FIDNet 将点云投影到 `64 x 2048` spherical range image。网络只处理每个像素中被保留的一个点，输出 2D prediction，再把标签回填到原始点云。对于投影时被遮挡或落入同一像素的点，NLA 在局部窗口内寻找 range 最接近的已预测点，将其 label 分配给该点。

## Loss / Post-processing / Training Strategy

- loss: 代码中组合 weighted cross entropy 和 Lovasz-Softmax，支持 top-k hard pixel mining，默认 `top_k_percent_pixels=0.15`。
- optimizer/training: 论文描述使用 Adam、one-cycle learning rate policy，batch size 2，训练 30 epochs。
- post-processing: NLA 只比较局部 patch 内 range 差异，不使用 KNN 的 Gaussian weighting 和 cutoff，因此更简单。
- README 环境: 推荐 `pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime` Docker，并提供依赖安装脚本。

## 实验数据集与指标

- SemanticKITTI test: 论文宣称在 `64 x 2048` 下优于当时 projection-based 方法。
- Validation ablation: CNN-only 为 55.4 mIoU / 11 ms；KNN 后处理为 58.7 mIoU / 2.7 ms；NLA 后处理为 58.9 mIoU / 1.2 ms。
- README 说明发布 checkpoint 在 validation set 可达到 58.8 mIoU。
- 论文补充更新提到新结构约 60.0 test mIoU、总参数约 6M、半精度下单帧约 0.01s。

## 优点

- decoder 完全无参数，结构简单，显存和部署压力小。
- 算子组合朴素：1x1 conv、ResNet、bilinear upsample、3x3 atrous conv，容易移植到常见推理框架。
- NLA 针对 projection many-to-one mapping，逻辑比传统 KNN 更直接，论文实验中更快且略优。
- 单 RTX 2080 Ti 11G 可训练和测试，门槛较低。
- 很适合作为研究 projection-based LiDAR segmentation 的轻量 baseline。

## 缺点

- 1x1 input/classification 设计局部建模能力有限，后续 CENet 证明 3x3 conv 能进一步提升精度和速度。
- 主要实验集中在 SemanticKITTI，数据集覆盖不如 CENet。
- NLA 假设局部 range 最近点标签可靠，在强遮挡、薄结构、密集重叠边界处仍可能传播错误标签。
- 代码路径和默认参数较脚本化，数据集路径、预训练权重和评测脚本需要手动调整。
- 相比后续方法，最终 mIoU 不是最强，适合作 baseline 而不是追求榜单精度。

## 工程复现风险

- 仓库 README 的数据集目录结构较固定，实际本地 SemanticKITTI 目录可能需要改脚本。
- 预训练权重存放路径要求具体目录名，直接运行前需要手动匹配。
- `if_KNN` 参数命名容易误导：`2` 实际表示作者的 NLA 后处理。
- 如果使用半精度或不同 PyTorch/CUDA 版本，速度和 mIoU 可能与论文略有差异。

## 适合借鉴的实现点

- FID fully interpolation decoding：多尺度特征直接 bilinear upsample + concat。
- 避免复杂 decoder，把参数和算力留给 backbone。
- NLA 后处理：针对投影遮挡点做最近 label assignment，而不是对所有点做 KNN voting。
- top-k hard pixel mining 与 Lovasz-Softmax 组合。
- 用最常见算子构建便于部署的 projection-based baseline。

## 与后续论文对比时应关注的维度

- 后续方法是否真正改进 backbone 表达，还是只增加 decoder/后处理复杂度。
- 后处理耗时是否计入总 FPS。
- 对边界和小目标的提升来自网络预测本身，还是来自投影回填策略。
- 是否保持 FID 的 parameter-free decoder 优势。
- 是否在相同 `64 x 2048` 输入分辨率下比较。
