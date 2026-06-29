# CENet 解读

## 一句话总结

CENet 是在 FIDNet 的 range-image 轻量框架上做的工程化增强：用更适合 GPU 的 3x3 卷积、更强激活函数和训练期辅助监督，在基本不增加推理开销的前提下提升 SemanticKITTI / SemanticPOSS 精度。

## 方法动机

FIDNet 已经证明了 projection-based LiDAR semantic segmentation 可以用很简单的结构取得不错速度和精度，但 CENet 作者认为 FIDNet 的 1x1 conv 更像逐点 MLP，没有充分利用 range image 的局部结构。CENet 的出发点是：range image 虽然不是普通 RGB 图像，但相邻像素仍有 LiDAR 扫描带来的局部几何关系，因此应当让输入模块和分类头具备局部感受野。

论文还强调实际部署中的算子效率。1x1 conv 的 FLOPs 和参数更少，但在 GPU 上计算密度不一定高；3x3 conv 可能有更好的硬件利用率。因此 CENet 不是单纯堆复杂模块，而是选择常见、成熟、容易部署的算子。

## 网络结构与关键模块

- 输入是投影后的 2D range image，通道通常为 `(x, y, z, d, r)`，其中 `d/range` 表示距离，`r/remission` 表示强度。
- 主干是 ResNet-style backbone，多层下采样得到 1, 1/2, 1/4, 1/8 分辨率特征。
- 解码思路继承 FIDNet：将多尺度特征用 bilinear interpolation 上采样回输入分辨率，然后 concatenate。
- 与 FIDNet 的关键区别是 CENet 默认使用 `modules/network/ResNet.py` 中的 3x3 input module 和 3x3 classification head；仓库也保留 `modules/network/Fid.py` 作为 FIDNet 风格 1x1 baseline。
- 训练配置默认 `pipeline: "res"`、`act: "Hardswish"`、`aux_loss: True`，说明最终主线是 3x3 卷积 + Hardswish + auxiliary heads。

## 输入/输出与投影表示

CENet 使用 spherical projection 将点云映射成 `H x W` range image，论文和配置中常见分辨率为 `64 x 512`、`64 x 1024`、`64 x 2048`。网络输出 2D label map，再投影回 3D 点。这个路线的优势是速度快、2D CNN 生态成熟；代价是投影会产生遮挡、多点映射到同一像素、远处小目标稀疏等问题。

## Loss / Post-processing / Training Strategy

- loss: weighted cross entropy 处理类别不均衡，Lovasz-Softmax 直接优化 IoU 相关目标，boundary loss 缓解边界模糊。
- auxiliary heads: 训练时对多尺度特征加辅助分割头，推理时可以移除，因此主要增加训练监督而非推理参数。
- activation: 论文实验比较 LeakyReLU、SiLU、Hardswish，最终配置多用 Hardswish。
- post-processing: 代码保留 RangeNet++ 风格 KNN 后处理，也有 `NN_filter` 辅助函数；论文重点不把后处理作为核心创新，而是强调网络本体的简洁高效。
- training: README 建议 SemanticKITTI 采用逐步分辨率训练，先 `64x512`，再加载预训练到 `64x1024`，最后到 `64x2048`。

## 实验数据集与指标

- SemanticKITTI test: 论文报告 `64 x 2048` 下 64.7 mIoU / 37.8 FPS，`64 x 1024` 下 62.3 mIoU / 67.9 FPS，`64 x 512` 下 60.7 mIoU / 84.9 FPS。
- SemanticPOSS: 论文报告相对 MINet 和 FIDNet baseline 有明显提升，特别强调小规模、稀疏场景下仍能保持优势。
- Ablation: 论文表 4 显示 baseline FIDNet 55.4 mIoU，加入 row/kernel/activation/auxiliary loss 后最高到 65.3 mIoU；表 5 显示 3x3 kernel 在 3 个分辨率下均比 1x1 kernel 更快。

## 关于相对 FIDNet 的额外 3.5 mIoU

论文表 4 中，官方 FIDNet code 是 55.4 mIoU；CENet 第 2 行在不启用 RowKS、SiLU/Hardswish、auxiliary loss 的情况下达到 58.9 mIoU，形成 3.5 mIoU 差距。论文原文说明该行是使用 CENet 作者自己的网络实现，并移除了 FIDNet 需要的 normal vector。因此，这 3.5 mIoU 不应简单归因到 kernel size、激活函数或辅助 loss，而更像是 CENet re-implemented FID-style baseline 的复合增益。

可能来源包括：

- 训练 pipeline 差异：CENet 使用自己的 parser、scheduler、训练策略和配置，不是逐行复现官方 FIDNet。
- loss 差异：CENet 的训练逻辑包含 weighted CE、Lovasz-Softmax 和 boundary loss；boundary loss 直接针对分割边界模糊，可能贡献了部分提升。
- 输入特征差异：FIDNet 官方默认使用 normal vector，CENet 表 4 第 2 行明确移除了 required normal vector。normal 特征可能带来额外噪声或训练不稳定，移除后改变了输入分布。
- 网络/head 实现差异：CENet 仓库中的 `modules/network/Fid.py` 是 CENet 框架下的 FID-style baseline，并不完全等同于官方 FIDNet 的 ASPP-like classification head 和脚本化训练实现。
- 评估口径差异：表 4 是 SemanticKITTI validation，输入 `64 x 512`，且直接在 projected 2D image 上评估，不等价于完整 3D point-wise evaluation 加后处理的官方 FIDNet 口径。

因此，严谨表述应为：CENet 的显式创新点之外，还存在一个 re-implemented baseline gain。若要确定来源，需要额外做消融：official FIDNet -> CENet data/training pipeline -> boundary loss -> remove normal -> CENet Fid.py architecture。

## 优点

- 改动非常工程友好：3x3 conv、Hardswish/SiLU、bilinear interpolation、auxiliary loss 都是成熟算子或训练技巧。
- 相比 FIDNet 明显提高 mIoU，同时保持较高 FPS，适合作为 range-image 分割的强 baseline。
- auxiliary heads 只在训练期使用，推理期不增加有效参数，适合部署。
- 支持 SemanticKITTI 和 SemanticPOSS，代码覆盖训练、推理、评测、可视化。
- README 明确提供预训练权重和日志入口，便于复现实验。

## 缺点

- 仍然受 projection-based 方法固有限制影响：遮挡、多点映射、稀疏远距离目标和垂直结构可能损失信息。
- 精度提升依赖多项训练策略叠加，复现时需要严格对齐分辨率阶段、activation、aux loss、scheduler 和 loss 权重。
- README 说明模型和 config 因多次更新可能存在不一致，工程复现需要排错。
- 相比 FIDNet，训练逻辑和配置更复杂；如果目标是极简实时部署，CENet 的工程面更重。
- 论文中的主要优势建立在 SemanticKITTI/SemanticPOSS，跨传感器或不同线数 LiDAR 的泛化仍需要额外验证。

## 工程复现风险

- 数据集路径、label yaml、训练分辨率阶段和 checkpoint 名称需要手动对齐。
- 代码保存文件仍使用 `SENet` 命名，和论文 CENet 命名不完全一致。
- CENet 训练期包含 boundary loss 和 auxiliary loss，显存和训练时间压力高于纯 FIDNet。
- 如果只想快速推理，建议优先使用作者提供的 pretrained model；如果从头训练，需要按 README 的 staged resolution strategy 复现。

## 适合借鉴的实现点

- 用 3x3 conv 替代 1x1 conv，获得更大局部感受野和更好的 GPU 算子效率。
- 保留 FIDNet 的 parameter-free interpolation decoding，避免复杂 decoder。
- auxiliary segmentation heads 作为训练期深监督，提高 backbone 学习质量。
- 使用 Hardswish/SiLU 作为低成本非线性增强。
- 多 loss 组合：weighted CE + Lovasz-Softmax + boundary loss。

## 与后续论文对比时应关注的维度

- 是否仍采用 range-image projection，还是融合 point/voxel 分支。
- 提升来自网络表达、后处理、蒸馏、数据增强还是多模态融合。
- 推理速度是否包含投影、网络、后处理、投影回点云的完整 pipeline。
- 参数量、FLOPs 和实际 FPS 是否在相同硬件/分辨率下比较。
- 对小目标、边界、动态物体、远距离目标的类别级 IoU 是否真正改善。
