# W2 设计待办（2×2 归因实验触发项，2026-08-14）

由 exactness 漂移归因实验的预授权路径（3b/3c）产生，W2 训练设计必须纳入：

## 1. off-policy 程度显式监控（预授权 3b）

训练分布评估时报告：**训练 token 中属 A100 teacher-forcing top-1 的占比**
（预期 ~96%，对应 pilot 漂移登记 95.93%），并分段报告：
- 按 rollout 内相对位置十分位（pilot 显示轻微前高后低：345→210/decile）
- 按样本长度桶（L<50 / 50-100 / 100-200 / ≥200）
- 按 token 类型（实词/子词延续/标点/数字——pilot：mismatch 富集实词 1.11×，
  标点 0.50×、数字 0.02× 贫化）
实现提示：全量 manifest 每条 records 已含 n_match/n_pos 与 mismatch 位置
（head16 截断），聚合无需重跑 forward。

## 2. OOD 风险登记（预授权 3c）

OOD 评估锚定 V100：推理时 ctx feature 由 target 在 V100 上现算，
与训练 feature（A100 抽取）存在同源数值漂移（幅度参考：TF 側硬件效应
Arm4vs3，AR 側 0.16%/位置）。预期 drafter 对 feature 输入扰动鲁棒；
**若 MM-Vet OOD 接受率异常，此项列为首查因素**。
可选的对照实验（届时再议，不预授权）：50 条 panel 在 V100 现算 feature
喂 drafter vs A100 缓存 feature 喂 drafter，比对块接受率差。

## 3. 沿袭事项

- anchor 数最终值（min(512, α×有效位置数)，α∈[1,2]）在 W2 代码审查时锁定
- 训练消耗 ctx_cache_35k 时按 manifest spans 限定 anchor 采样范围于 rollout 段
- 参考资料先读：dspark-aeon-27b 社区 recipe、DFlare arXiv:2606.02091
  （多 anchor 稀疏 mask 实现参照）
