# CENet vs FIDNet 对比解读

## 总体定位

FIDNet 是一个极简 projection-based LiDAR semantic segmentation baseline，核心是 parameter-free FID 解码和 NLA 后处理。CENet 可以看作在 FIDNet 基础上的增强版本：保留多尺度插值融合的主线，但重新设计输入/分类卷积、激活函数、辅助监督和 loss，使模型表达能力更强。

如果目标是快速复现一个轻量 pipeline，FIDNet 更合适；如果目标是拿一个仍然简洁但精度更强的 range-image baseline 做后续改进，CENet 更合适。

## 主要不同点与影响

| 维度 | FIDNet | CENet | 带来的差异 |
| --- | --- | --- | --- |
| 输入模块 | 主要用 1x1 conv，把每个 range-image pixel 当作点特征处理 | 改用 3x3 conv，利用 range image 局部邻域 | CENet 局部几何建模更强；对边界、薄结构、相邻点上下文更友好。FIDNet 更像逐点 MLP，结构更轻但表达受限。 |
| backbone/encoder 表达 | ResNet-34 风格多尺度特征 | ResNet-style 结构，并围绕 3x3 conv 和激活函数做优化 | CENet 的 backbone 输入特征更充分，整体 mIoU 更高；FIDNet 更适合做干净 baseline。 |
| decoder | FID: bilinear interpolation 上采样多尺度特征并拼接，无参数 | 基本继承 FID 思路，多尺度插值拼接仍是核心 | 二者都避免复杂 decoder，速度和部署友好；差异主要来自 CENet 解码前后的特征增强。 |
| classification head | ASPP-like atrous conv + 1x1 conv 输出 | 3x3 classification head + segmentation head | CENet 分类头具备更强局部融合能力；FIDNet 的 head 更强调简洁和低复杂度。 |
| 大卷积核/局部上下文 | 不强调，输入模块偏 1x1 | 明确用 3x3 替代 1x1，并论证 GPU 上可能更快 | CENet 在不显著增加推理负担的情况下提升局部上下文表达；对 car boundary、pole、traffic-sign 等局部细节更有潜在收益。 |
| activation | LeakyReLU 为主 | SiLU / Hardswish 可选，默认配置多用 Hardswish | CENet 非线性表达更强，论文 ablation 显示能提升 mIoU；代价是训练复现变量更多。 |
| auxiliary head / multi-loss | 主要 weighted CE + Lovasz-Softmax，支持 top-k hard mining | weighted CE + Lovasz-Softmax + boundary loss，并引入多尺度 auxiliary heads | CENet 训练监督更强，收敛和边界质量更好；但训练过程更复杂，超参数更多。 |
| 后处理 | 重点提出 NLA，替代 KNN，处理 many-to-one mapping | 保留 KNN/NN filter 相关代码，论文重心在网络增强 | FIDNet 后处理思想更鲜明、更适合研究投影回填；CENet 更依赖网络本身的表达能力提升。 |
| 数据集支持 | 主要 SemanticKITTI | SemanticKITTI + SemanticPOSS | CENet 代码和论文覆盖更多数据集，泛化论据更充分；FIDNet 更聚焦单一 benchmark。 |
| 代码成熟度 | 脚本直接、结构较轻，但路径和参数较硬编码 | 功能更全，有训练/推理/评测/可视化和多配置，但 README 提醒配置可能不一致 | FIDNet 更容易读懂核心思路；CENet 更适合完整实验，但复现时需要处理配置细节。 |

## 精度、速度与资源差异

- 精度: CENet 明显更强。论文中 FIDNet baseline validation 为 55.4 mIoU，CENet ablation 最优到 65.3 mIoU；SemanticKITTI test 上 CENet `64 x 2048` 报告 64.7 mIoU。
- 额外 baseline 增益: CENet 表 4 中，即使不启用 kernel size、activation、auxiliary loss，作者自己的基础网络也从官方 FIDNet 的 55.4 提到 58.9 mIoU。这个 3.5 mIoU 更应视为 re-implemented FID-style baseline gain，可能来自训练 pipeline、boundary loss、移除 normal vector、网络/head 实现和 2D validation 评估口径差异，而不是某个单独模块。
- 速度: 二者都很快。FIDNet 的优势是 decoder 无参数、NLA 后处理轻；CENet 证明 3x3 conv 在实际 GPU 上不一定比 1x1 慢，论文报告 `64 x 2048` 为 37.8 FPS。
- 参数/算子: FIDNet 结构更朴素，部署风险更低；CENet 使用的也都是常见算子，但训练期 auxiliary heads 和多 loss 会增加训练复杂度。
- 边界与小目标: FIDNet 从后处理角度解决投影回填导致的边界模糊；CENet 从特征表达和 boundary loss 角度增强边界质量。FIDNet 更适合分析 projection artifact，CENet 更适合追求最终分割质量。

## 3.5 mIoU Baseline Gap 的解释

CENet 论文表 4 的第 1 行是官方 FIDNet code，55.4 mIoU；第 2 行是 CENet 作者自己的网络实现，58.9 mIoU。第 2 行还没有启用 RowKS、SiLU/Hardswish、Plan A/Plan B auxiliary loss，因此这部分差距不能归入 CENet 后续显式模块。

更合理的拆解是：

- 训练和数据管线不同：CENet 的 parser、训练脚本、scheduler、配置组织和评估流程均不同于官方 FIDNet。
- loss 组合不同：CENet 引入 boundary loss，与 weighted CE 和 Lovasz-Softmax 共同监督；这可能改善边界相关类别和投影上采样带来的模糊。
- 输入特征不同：CENet 论文说明该 baseline 移除了 FIDNet 所需的 normal vector，输入从依赖 normal 的设置变为更简洁的 range-image 特征。
- baseline 实现不同：CENet 的 `Fid.py` 是 CENet 框架下的 FID-style 结构，不是官方 FIDNet 的完全同构实现。
- 评估口径不同：该 ablation 使用 `64 x 512`，并直接在 projected 2D image 上评估，不能直接等同于 FIDNet 论文的完整 3D 点级结果。

因此，在后续引用时建议把这部分称为 “CENet re-implemented baseline gain”。如果要严格归因，需要独立复现实验逐项控制 official FIDNet、CENet 训练管线、boundary loss、normal vector 和 head 结构。

## 优劣结论

- FIDNet 的最大价值是方法论清晰：range-image 分割不一定需要复杂 decoder，bilinear interpolation + concat 就能作为高效解码器。
- CENet 的最大价值是证明在 FIDNet 框架上，简单替换为 3x3 conv、改激活函数、加辅助监督，就能显著提升精度而不破坏实时性。
- 工程 baseline 选择:
  - 追求最小结构、快速读懂、部署验证: 选 FIDNet。
  - 追求更强精度、更多数据集支持、后续继续改进: 选 CENet。
- 后续改进起点:
  - 如果研究 projection 后处理、遮挡点标签回填、range-image artifact，FIDNet 更直接。
  - 如果研究 backbone、轻量上下文建模、训练监督、蒸馏或鲁棒性，CENet 更合适。

## 建议的后续实验

- 在同一环境下统一 `64 x 2048` 输入，分别跑 FIDNet、CENet fid pipeline、CENet res pipeline，拆分网络耗时和后处理耗时。
- 记录 car、bicycle、motorcycle、person、pole、traffic-sign 等类别级 IoU，看提升是否主要来自小目标/边界类别。
- 单独比较 KNN、NLA、无后处理在 CENet 输出上的效果，判断 CENet 是否仍需要 FIDNet 的 NLA 思路。
- 做一次跨数据集验证，例如 SemanticKITTI 训练到 SemanticPOSS 或其他自采 LiDAR，观察 3x3 局部建模是否比 1x1 更稳。
