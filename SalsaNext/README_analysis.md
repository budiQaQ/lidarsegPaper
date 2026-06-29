# SalsaNext 解读

## 一句话总结

SalsaNext 是面向自动驾驶 LiDAR 语义分割的 range-image encoder-decoder 网络，在 SalsaNet 基础上通过上下文模块、膨胀残差卷积、pixel shuffle decoder 和不确定性估计提升精度与可靠性。

## 方法动机

自动驾驶语义分割不仅需要实时和高精度，还需要知道模型在哪些点上不可靠。SalsaNext 选择 projection-based 路线，把 LiDAR 投影为 2D range image，以获得 2D CNN 的高吞吐；同时加入 Bayesian uncertainty，为后续规划/决策提供置信度线索。

## 网络结构与关键模块

- context module: 输入后先经过多层 residual dilated context block，扩大 360 度 LiDAR scan 的上下文感受野。
- encoder: 用 residual dilated convolution stack 替代 SalsaNet 的普通 ResNet block，包含不同 dilation/kernel 分支并拼接。
- downsample: 使用 average pooling 替代 stride convolution，减轻模型并降低下采样副作用。
- decoder: 使用 pixel shuffle 替代 transposed convolution，避免 checkerboard artifacts。
- skip fusion: decoder 与 encoder 的中间特征拼接，恢复空间细节。
- central dropout: dropout 不作用于最前和最后层，尽量保留基础边缘/曲线特征和最终输出稳定性。
- uncertainty branch: `SalsaNextAdf.py` 使用 ADF/Bayesian treatment 估计 epistemic 和 aleatoric uncertainty。

## 输入/输出与投影表示

输入是投影后的 5 通道 range image，代码中 `SalsaNext` 使用 `ResContextBlock(5, 32)`。输出是 2D semantic probability map，再投影回原始点云。与 CENet/FIDNet 一样，SalsaNext 需要处理 projection loss、遮挡点和 KNN 后处理问题。

## Loss / Post-processing / Training Strategy

- loss: weighted cross entropy + Lovasz-Softmax，直接优化类别不均衡和 IoU。
- uncertainty 版本额外使用 heteroscedastic loss。
- optimizer: 代码中使用 SGD + warmupLR。
- post-processing: 代码包含 RangeNet++ 风格 KNN 后处理。
- README 提供 `train.sh` 和 `eval.sh`，支持 `-u` 启用 uncertainty 版本。

## 实验数据集与指标

论文主要在 SemanticKITTI 上评估，定位是 fast, uncertainty-aware semantic segmentation。论文中与 RangeNet++、SalsaNet 等 projection-based 方法比较，并强调精度、运行时间和参数量的平衡。README 提供 SemanticKITTI 训练/评测流程和预训练模型入口。

## 优点

- 比 PointNet/PointNet++ 更适合完整自动驾驶 LiDAR scan 的实时语义分割。
- encoder-decoder 结构比 FIDNet 更复杂，但能获得更强上下文建模能力。
- pixel shuffle decoder 避免转置卷积棋盘伪影，适合密集 2D prediction。
- uncertainty 输出是实际自动驾驶系统中的重要附加价值。
- 代码基于 RangeNet++ 框架，训练、评估、可视化和 KNN 后处理较完整。

## 缺点

- 仍然是 projection-based 方法，存在投影遮挡、多点映射和远距离稀疏问题。
- 网络比 FIDNet/CENet 更复杂，部署和 TensorRT 优化难度更高。
- uncertainty 版本增加训练/推理复杂度，且实际能否提升下游安全决策需要额外验证。
- 主要验证集中在 SemanticKITTI，跨传感器泛化仍需实测。

## 工程复现风险

- 需要准备 SemanticKITTI 目录与 label yaml。
- uncertainty 版本需要额外依赖和 Monte Carlo/ADF 相关流程。
- 预训练权重在 Google Drive，未默认下载。
- KNN 后处理耗时是否计入总 FPS 需要复现实验时明确记录。

## 适合借鉴的实现点

- range-image encoder-decoder 的强上下文建模。
- residual dilated convolution stack。
- pixel shuffle 用于 LiDAR range image decoder。
- weighted CE + Lovasz-Softmax。
- uncertainty-aware semantic segmentation 设计。

## 与后续论文对比时应关注的维度

- 精度提升来自 encoder-decoder 表达还是后处理。
- uncertainty 是否只是可视化输出，还是能用于拒识/规划。
- 与 FIDNet/CENet 的差异：复杂 decoder vs parameter-free FID。
- 与 PointNet/PointNet++ 的差异：projection 高吞吐 vs raw point 几何保真。
