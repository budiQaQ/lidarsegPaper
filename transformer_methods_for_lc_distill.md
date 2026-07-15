# Transformer 方法与 Range-View LC 蒸馏借鉴索引

## 分层结论

| 层级 | 方法 | 与当前方向的关系 | 参数量 | 计算耗时/速度 | 最值得借鉴的点 |
| --- | --- | --- | --- | --- | --- |
| 直接相关 | RangeFormer | LiDAR-only range-view 强基线 | 约 23.7M | PandaSet validation 约 54 ms | many-to-one / semantic incoherence / shape deformation 问题定义，RangePost，STR |
| 直接相关 | RangeViT | range image + ViT | 未确认，取决于 ViT backbone | 未找到可横比 latency | image-pretrained ViT 迁移、非方形 patch token、range-view token KD |
| 直接相关 | RangeRet | range image + Retentive Network | 约 3.8M | PandaSet validation 约 38 ms | Circular Retention、轻量长程建模、水平 360 度环绕连续性 |
| 直接相关 | FLARES | multi-range range-view pipeline | 取决于 backbone；FLARES-FIDNet/CENet 约 6.0M/6.8M | 约 19 ms / 22 ms；论文称超过 40% speed-up | multi-range low-resolution student、WPD+/MCF、NNRI 后处理 |
| 模块参考 | Swin Transformer | 通用 2D backbone | Swin-T/S/B 约 28M/50M/88M；UPerNet 约 60M/81M/121M | ImageNet 分类约 755/437/278 FPS；分割 teacher 需重测 | shifted window attention、多尺度 feature、camera teacher |
| 3D 对照 | Point Transformer V3 | 现代 3D point Transformer | 未确认，随 Pointcept 配置变化 | 整理中记录约 44 ms | serialized point tokens、FlashAttention、3D teacher 或 sparse student 对照 |
| 3D 对照 | Stratified Transformer | point-based long-range context | 未确认 | 未找到可横比 LiDAR scan latency | local dense + distant sparse attention，可迁移为 range-view 稀疏全局注意力 |

## 对 LC 蒸馏最有价值的设计

1. Range-view student 不建议直接做全局 ViT。`64 x 2048` token 数过大，应该参考 Swin/RangeViT，用局部窗口或非方形 patch。
2. Camera teacher 到 range-view student 的 KD 可以按 token/window 做，而不是只做 point-wise L2。
3. RangeFormer 提醒必须处理投影问题：depth discontinuity、many-to-one aliasing、semantic incoherence 都应进入 mask 或 loss 权重。
4. RangeRet 提供了比 ViT 更轻的长程上下文方案，CiR 的 circular distance 很适合约束 range-view token KD。
5. FLARES 提供了 multi-range low-resolution student 的训练范式，适合把 camera teacher feature 分配到多个 sub-cloud/range token，而不是挤到单张 `64 x 2048` 全景图。
6. PTv3/Stratified 适合作为 3D teacher 或对照，不适合作为轻量 range-view student。
7. STR/FLARES 都适合 LC 蒸馏训练：camera branch + range student 同训显存会增加，低宽度训练可降低成本。

## 推荐实现路线

第一版不要重写整个 CENet。建议：

- Student: CENet 或 FIDNet。
- 插件: bottleneck 加轻量 shifted-window block 或 RangeRet-style Circular Retention block；数据 pipeline 可借鉴 FLARES 的 multi-range splitting。
- Teacher: image-pretrained Swin/SegFormer/DINOv2 或 RangeViT-style ViT feature。
- 对齐: camera feature -> LiDAR point -> range pixel/token；如果采用 FLARES，则进一步对齐到 sub-cloud token。
- Loss:
  - logit KD: camera-assisted fusion logits 蒸馏给 LiDAR-only logits。
  - feature KD: window/token-level feature KD。
  - boundary KD: 只在 depth discontinuity / image edge / semantic boundary 附近增强。
  - class-aware KD: person、bicyclist、motorcyclist、pole、traffic-sign 加权。

## 与已整理论文的关系

- 2DPASS 提供 training-only camera prior 的范式。
- RangeViT 提供 range-view ViT student 的结构参考。
- RangeFormer 提供 range-view 强基线和投影问题处理框架。
- RangeRet 提供轻量 Retentive Network student 的结构参考。
- FLARES 提供 multi-range 低分辨率 student、WPD+/MCF 增强和 NNRI 回投影参考。
- Swin 提供 camera teacher 和轻量窗口 attention。
- PTv3/Stratified 提供 3D Transformer 对照，帮助解释为什么仍选择 range-view student：更规则、更易部署、更适合低延迟。
