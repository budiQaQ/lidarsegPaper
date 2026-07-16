# 96 x 480 x 3 多回波点云的单层输出方案

## 问题定义

输入是一帧按 range-view 组织的多回波点云：

```text
H = 96
W = 480
K = 3，每个像素最多保存 3 个三维点
```

统一张量定义：

```python
points:     [B, K, F, H, W]  # xyz/range/intensity 等点特征
ranges:     [B, K, H, W]     # 按距离从近到远排序
valid_mask: [B, K, H, W]     # 候选点是否存在
```

最终任务不是为全部多层点输出语义标签，而是每个 `(u,v)` 位置最多输出一个原始候选点：

```text
0: 当前像素不输出点
1: 输出 layer 0
2: 输出 layer 1
3: 输出 layer 2
```

模型不回归新的 xyz，而是选择一个原始测量点。这能避免加权坐标产生不存在于传感器测量中的虚假点。

## 三种方法的公共特征提取

为保证对比公平，三种方法共用同一套特征提取结构，只改变分类头、训练目标和输出决策。

### 1. 候选点共享编码

对每个 layer 的点特征使用同一个 `PointStem`：

```text
p0 --shared PointStem--> e0
p1 --shared PointStem--> e1
p2 --shared PointStem--> e2
```

代码中将 `[B,K,F,H,W]` 重排为 `[B*K,F,H,W]`，只调用一份 `1x1 Conv + GroupNorm` 参数，再恢复为 `[B,K,Dp,H,W]`。这一步编码 xyz、range、intensity 等单点属性，不负责最终选择。

### 2. 多层几何描述

在每个像素构造：

```text
e0, e1, e2
mask0, mask1, mask2
count / 3
range0, range1, range2
range1-range0, range2-range1
```

range 先按 `range_scale` 归一化，无效点及无效 depth gap 填 0，并始终保留 mask 区分真实零值与 padding。

### 3. CENet-style 联合上下文

三层候选特征和几何特征在 channel 维拼接，只输入一个 CENet-style encoder-decoder：

```text
[e0,e1,e2,geometry]
          |
          v
large-kernel encoder-decoder
          |
          v
G: [B,Dc,H,W]
```

`G` 是公共场景上下文，用来学习多层现象是否在目标区域连续出现、当前点是否与周围几何一致，以及孤立回波是否更像噪声。代码使用 GroupNorm，避免稀疏 layer 和小 batch 导致 BatchNorm 统计不稳定。

### 4. 候选点与公共上下文融合

对第 `k` 个候选点构造：

```text
[e_k, G, normalized_range_k, range_k-range_0, valid_k, rank_k]
```

使用另一个三层共享的融合器得到 `q_k`。因为 `q_k` 同时含有自身点特征和三层联合上下文，模型不会像纯 batch 拆层那样丢失同像素的多层关系。

## 方法一：候选二分类 + 最近目标规则

### 分类逻辑

共享 binary head 分别输出：

```text
z0 = target score of layer 0
z1 = target score of layer 1
z2 = target score of layer 2
```

经 sigmoid 和阈值得到每个候选点的 target/noise 判断，然后在被判为 target 的点中选择 range 最小者。如果没有 target，输出 `none`。

```text
candidate logits -> sigmoid -> threshold -> target mask
target mask + ranges -> nearest valid target -> single-layer point
```

### 训练监督

需要 `[B,K,H,W]` 候选点二值标签，仅在 `valid_mask=True` 的位置计算 focal loss。

### 特点

- 优点：监督直接、可解释、小数据下较稳定。
- 缺点：需要人工阈值；最近点规则与模型分开优化；不能根据上下文主动选择更远但更可信的候选点。

## 方法二：K+1 联合选择

### 分类逻辑

对 `q0/q1/q2` 分别使用共享 candidate selection head，得到三个候选分数。公共上下文 `G` 经 none head 得到不输出分数：

```text
[s_none, s_layer0, s_layer1, s_layer2]
                   |
                   v
              4-way softmax
```

无效 layer 的 logit 被 mask 为极小值，`none` 始终可选。模型可以选择非最近层，最终 `argmax` 直接得到 `0..K` 选择索引。

### 训练监督

需要 `[B,H,W]` 选择标签：

```text
0 = none
1 = layer 0
2 = layer 1
3 = layer 2
```

使用带类别权重的 cross entropy。必须单独处理 `none` 和 layer 0 的样本占比过高问题，否则模型容易学会捷径。

### 特点

- 优点：训练目标与最终单层输出一致；不需要 sigmoid 阈值或最近点规则；三个候选点直接竞争。
- 缺点：选择标签必须唯一且可靠；容易偏向 `none/layer0`；只观察最终选择时，难以解释其他候选点为什么被拒绝。

## 方法三：K+1 联合选择 + 候选二分类辅助监督

### 分类逻辑

主分支与方法二相同，最终输出仍完全由 `K+1` 联合选择头决定。另外对 `q0/q1/q2` 增加共享 binary head，监督每个候选点是 target 还是 noise：

```text
candidate features
       |---------------- selection head --> none/layer0/layer1/layer2
       |
       +---------------- binary head ----> target/noise per candidate
```

推理时不使用 binary head 执行阈值或最近点后处理，binary head 只用于训练约束、调试和可解释性。

### 训练监督

```python
loss = selection_cross_entropy + 0.2 * candidate_focal_loss
```

需要同时提供最终选择标签和每个候选点的 target/noise 标签。

### 特点

- 优点：直接优化最终选择，同时防止模型只记住 layer 分布；binary head 能显示每个候选被保留或拒绝的原因。
- 缺点：标注和损失更复杂；辅助损失权重需要消融；候选标签和最终选择标签矛盾时会产生梯度冲突。

## 三种方法对比

| 维度 | 二分类+最近点 | K+1 联合选择 | 联合选择+辅助二分类 |
|---|---|---|---|
| 最终输出是否直接优化 | 否 | 是 | 是 |
| 是否需要人工阈值 | 是 | 否 | 否 |
| 是否强制最近点 | 是 | 否 | 否 |
| 候选点可解释性 | 强 | 弱 | 强 |
| 训练稳定性 | 高 | 中等 | 中高 |
| 标注需求 | 候选二值标签 | 最终选择标签 | 两类标签都需要 |
| 走捷径风险 | 低 | 高 | 中等 |
| 建议定位 | 强 baseline | 主任务 baseline | 推荐最终方案 |

## 完整代码

完整 PyTorch 实现位于 [models.py](./models.py)，包含：

- `MultiReturnEncoder`: 候选共享编码、几何特征、CENet-style 上下文和候选融合。
- `BinaryNearestModel`: 方法一。
- `JointSelectionModel`: 方法二。
- `JointSelectionAuxModel`: 方法三。
- `masked_binary_focal_loss`: 候选点 masked focal loss。
- `selection_cross_entropy`: `K+1` 选择损失。
- `joint_selection_aux_loss`: 方法三联合损失。
- `gather_selected_points`: 将 `0..K` 选择索引转换为单层点图。

该 context backbone 是可独立运行的 CENet-inspired 实现，不是官方 CENet 代码的逐行复制。如果工程中已有 CENet encoder-decoder，可以仅替换 `CENetStyleContextBackbone`，其他输入输出接口保持不变。

## 使用方法

### 创建模型

```python
from models import build_model

model = build_model(
    method="joint_selection_aux",
    point_channels=5,
    num_layers=3,
    point_feature_channels=32,
    context_channels=64,
    candidate_channels=64,
    backbone_base_channels=32,
    range_scale=100.0,
)
```

### 训练方法三

```python
from models import joint_selection_aux_loss

outputs = model(points, ranges, valid_mask)

losses = joint_selection_aux_loss(
    outputs=outputs,
    selection_labels=selection_labels,  # [B,H,W], 0..3
    candidate_labels=candidate_labels,  # [B,3,H,W], 0/1
    valid_mask=valid_mask,
    selection_class_weights=class_weights,
    auxiliary_weight=0.2,
)

losses["loss"].backward()
```

### 输出单层点云

```python
model.eval()
prediction = model.predict(points, ranges, valid_mask)

single_layer_map = prediction["selected_points"]
# [B,F,96,480]

output_mask = prediction["output_mask"]
# [B,96,480], False 表示当前像素不输出点

selection = prediction["selection"]
# 0=none, 1=layer0, 2=layer1, 3=layer2
```

如果需要紧凑的 `N x F` 点云：

```python
compact_points = single_layer_map.permute(0, 2, 3, 1)[output_mask]
```

## 标签构建注意事项

- `selection_labels` 必须表示最终期望输出，而不是简单复制“最近点”规则，否则联合选择模型只会学会复制后处理。
- 如果多个候选点都是合法输出，必须确定唯一的业务优先级，或使用 soft target，否则 hard cross entropy 会惩罚同样合理的选择。
- 无效 layer 不得作为 selection label，候选二分类损失也必须由 `valid_mask` 屏蔽。
- 建议同时统计 `P(selection | count)`、`P(count | target)` 和各 layer 选择频率，确认模型没有退化为恒选 `none/layer0`。

## 建议消融实验

| 实验 | 目的 |
|---|---|
| 原始 CENet 只使用 layer 0 | 单回波 baseline |
| 方法一 | 验证候选二分类和确定性规则 |
| 方法二 | 验证直接优化最终选择的收益 |
| 方法三 | 验证辅助二分类是否抑制捷径 |
| 移除 geometry | 验证 count/range gap 的贡献 |
| 移除 context backbone | 验证局部目标形状的贡献 |
| 移除 rank 特征 | 检查模型是否过度依赖 layer 编号 |

评估不应只看总 accuracy，至少应报告：

```text
target retention recall
noise removal precision/recall
none/layer0/layer1/layer2 selection accuracy
single-return / double-return / triple-return accuracy
distance-binned accuracy
output point count error
latency and parameter count
```

## 与 CENet / FLARES 的关系

- 借鉴 CENet 的大卷积核上下文和高效 encoder-decoder，但将分类目标改为多候选选择。
- 借鉴 FLARES 对 multi-range 和 range difference 的利用，但不再进行 sub-cloud splitting，因为输入已经显式保存最多三个重叠点。
- 不使用 NNRI 恢复全部点，因为任务目标是选择单层输出，而非为每个原始点分配语义标签。

参考：

- CENet: https://arxiv.org/abs/2207.12691
- FLARES: https://arxiv.org/abs/2502.09274
