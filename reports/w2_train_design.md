# W2 训练循环设计 — block diffusion drafter（Qwen2.5-VL-7B target）

2026-08-14。状态：**待审查**。代码实现引用本文档节号。
数据：ctx_cache_35k（全量抽取 176267512 在途；开发/测试只读 pilot 分片）。

## 0. 前置阅读对表笔记

### 0.1 dspark-aeon-27b（commit 6499f82，train/train_head.py + train/dflash.py）

与 DFlash 论文的出入点（记录，不照搬）：

| 项 | dspark 做法 | DFlash 论文 / 我们 |
|---|---|---|
| anchor 批组装 | **每 anchor 一行**，ctx 左 padding 复制，行间 batch | 论文/DFlare：多 anchor **打包单序列** + 稀疏 mask（ctx K/V 每序列只算一次）。dspark 的 target 是 27B、head 极小，ctx 复制可承受；我们 drafter 1.23B + 150-300 anchor/序列，必须打包 |
| 块内噪声位 | mask_token（有 teacher-forced future 可选分支） | 同论文（mask），我们不启用 TF-future 分支 |
| loss | CE 默认 + KD 变体（forward KL/L1/acceptance），**γ=0 均匀为默认** | 论文 Eq.4 指数衰减 γ=7（block 16）为主配置 |
| label 对齐 | "index p−1 predicts x_{a+p}"（block 槽 m 预测 anchor+m） | 一致 ✓（与推理 `[:, 1−block_size:]` 对应） |
| causal_head 选项 | 支持块内因果注意力变体 | 论文为块内双向；不采用 |
| mask_token_id | 248070（其 27B 词表的保留位） | 我们选 151662（见 §1a） |

有用的实现细节直接借鉴：anchor 合法域推导（a≥ctx 首、块内至少 1 个可监督位）、
padding 位零权重、`block_weights` 公式形态。

### 0.2 DFlare arXiv:2606.02091 §3.3

- 证实：全部采样块**拼接为单序列**，稀疏 mask 联合处理；块内双向 + 块间禁止 +
  各块只见"对应的" target ctx features——与本设计 §1c 三规则一致
- 位置 id 与 FlexAttention 实现细节论文未展开（我们自行规范，§1c/§1d）
- 进阶 γ 调度：γ(s) = γ₀ + (s/S)(γ_max−γ₀)（γ₀=4.5 逐 epoch +1）。
  **不采用**（单变量纪律：首训用 DFlash 固定 γ；DFlare 调度列为 W3+ 消融候选）

### 0.3 DFlash arXiv:2602.06036 §4.2 + Appendix A.1 逐句映射表

| 论文条目 | 内容 | 我们实现的落点 |
|---|---|---|
| KV injection | target 特征融合后注入 drafter 各层 K/V，全程 conditioning | 复用 `external/dflash` `DFlashDraftModel.fc/hidden_norm` + 各层 `k_proj/v_proj(target_hidden)`（zero 重写，§2 组装） |
| 随机 anchor | "randomly sample anchor tokens from the response, use each as first position of a block, mask the remaining"；App.A.1: 512/序列 | `ctx_dataset.sample_anchors`：均匀无放回，域=rollout span（§1b）；数量 min(512, α·L) 修订（W1 已批） |
| 跨块隔离 | 块内双向 + 可见对应 ctx；跨块禁止（sparse mask） | `ctx_dataset.build_block_mask`（§1c，双路径） |
| loss 加权 | w_k = exp(−(k−1)/γ)，**Eq. 4**；γ: block16→**7**, block10→5, block8→4 | `block_weights(B, γ=7)`（§1e） |
| 优化器 | AdamW lr **6e-4**，cosine，warmup ratio **0.04**，6 epochs，clip 1.0，max seq 3072 | `TrainConfig` dataclass（§1e）；max seq 对我们不约束（T≤~821） |

## 1. 设计决策

### 1a. mask token = `<|fim_pad|>` (id **151662**)

Qwen2.5-VL 词表无原生 [MASK]。候选为 6 个非 chat/非 vision 功能的保留特殊
token（151659-151664：fim_prefix/middle/suffix/pad、repo_name、file_sep），
已实测在 pilot 500 条完整 ids（vision+prompt+rollout）中**全部零出现**；
数据中实际出现的特殊 token 仅 {im_start, im_end, vision_start, vision_end,
image_pad}。选 `<|fim_pad|>`：
- id 151662 < EFFECTIVE_VOCAB 151936 → embedding 行真实存在（冻结的 target
  embed 提供一个固定、与文本 token 可区分的向量，drafter 从零学其含义）
- 语义本身即"填充占位"，用作 FIM 填充符，图文对话分布中零出现
- 非 chat template 功能符（im_*/vision_* 均在用，不可占用）
- 运行时守卫：dataset 加载每条样本时 assert `mask_id ∉ ids`（全量数据的
  零出现验证内嵌于训练数据管线，非一次性抽样）

### 1b. anchor 采样与打包布局

- **域**：严格限于 rollout span `[P, T)`（读分片 `spans[4:6]`）；合法 anchor
  序列位 p ∈ [P, T−2]（p=T−1 无可监督位，排除）
- **数量**：`K = min(512, ceil(α·L))`，L=rollout 长度；α 候选 {1.0, 1.5, 2.0}，
  **默认 α=1.5**（理由：α=1 期望每个 rollout 位被覆盖 ~1 次但方差大；α=2 对
  L=256 的截断样本到 512 顶格、计算×2；1.5 在覆盖率（~78% 位置至少一个 anchor）
  与算力（中位 K≈216）间平衡。最终值本审查锁定）
- 均匀无放回；K > 合法域大小时取全域（短样本全覆盖）
- **打包布局**（每序列一行）：

```
keys:    [ ctx_0 … ctx_{T-1} | blk_1(16) | blk_2(16) | … | blk_K(16) | (noise pad) ]
queries: [                     blk_1(16) | blk_2(16) | … | blk_K(16) | (noise pad) ]
blk_k = [ x_{p_k}, m, m, …, m ]   （anchor 真 token + 15 个 mask=151662）
position_ids(noise) = p_k, p_k+1, …, p_k+15   （绝对扁平位，与推理一致）
position_ids(ctx)   = 0 … T−1
```

批维=序列；ctx 右 padding 到 batch 内 T_max（mask 屏蔽 pad 键），噪声段右
padding 到 K_max·B（loss 权重 0 + mask 全屏蔽）。ctx fc/K/V 每序列每层一次。

### 1c. 注意力 mask：三规则 + 双路径

规则（DFlash/DFlare 语义，噪声 query q 属块 k、anchor 位 p_k）：
1. **块内双向**：q 可见本块全部 16 个噪声键
2. **块间隔离**：q 不可见任何其他块的噪声键
3. **ctx 可见（因果于块粒度）**：q 可见 ctx 键位 c 当且仅当 **c < p_k**
   （严格小于——anchor 自身的 ctx 特征不可见，见 §1d 推导）

3 块示意（ctx 长 6，B=3，anchor 位 p=2,3,4；行=query，列=key；■=可见）：

```
            ctx0 ctx1 ctx2 ctx3 ctx4 ctx5 | b1a b1m b1m | b2a b2m b2m | b3a b3m b3m
blk1(p=2) q  ■    ■    ·    ·    ·    ·   |  ■   ■   ■  |  ·   ·   ·  |  ·   ·   ·
blk2(p=3) q  ■    ■    ■    ·    ·    ·   |  ·   ·   ·  |  ■   ■   ■  |  ·   ·   ·
blk3(p=4) q  ■    ■    ■    ■    ·    ·   |  ·   ·   ·  |  ·   ·   ·  |  ■   ■   ■
```

- **主路径（A100）**：FlexAttention `mask_mod(b,h,q_idx,kv_idx)` 闭包读
  (块归属表, anchor 位表)——布尔逻辑与上表逐元素等价
- **回退/交叉验证路径**：4D additive mask (1,1,Q,KV) + sdpa（旧线 probe_4d_mask
  已验证 sdpa 接受任意 4D mask）。两路径在相同输入下 logits 必须一致：
  CPU 单测用 sdpa-vs-eager 先验证 mask 本身，Flex-vs-sdpa 等价留 smoke 上卡验
- 掉卡兼容：V100 只有 sdpa 路径（FlexAttention 需 sm80+），与 B3c 结论一致

### 1d. ctx 对齐规范（TARGET convention，历史事故高发区）

**基本事实**：缓存 `ctx[ℓ, p, :]` = target 第 ℓ 抽取层在序列位 p 的输出态
（读完 ids[p] 之后）。TARGET convention：`argmax(lm_head(h_final[p])) = ids[p+1]`
——位 p 的特征"孕育"位 p+1 的 token。

**推理侧接线**（dflash_generate 逐行核对）：block 起点 start=t 时，
draft cache 恰 crop 到 t（`crop(start)`），即 ctx 覆盖位 **[0, t)**；anchor
token x_t 以 noise embedding 身份进入块首，其 ctx 特征**不**在可见集内
（它在本轮 verify 后才产出）。

**训练侧公式**（与推理严格同构）：anchor 序列位 p（rollout 偏移 j，p = P+j）：
- ctx 可见集：位 c ∈ **[0, p)** 的特征（mask 规则 3 的 c < p_k 即此）
- 块槽 m ∈ {0,…,B−1} 在序列位 p+m；槽 0 = anchor（无 label）
- 槽 m ≥ 1 的 label = `ids[p+m]`，若 p+m ≤ T−1；否则 −100（序列尾不足一块）
- drafter 输出槽 m 的 hidden → lm_head → 监督 `ids[p+m]`——正是推理中
  `draft_logits[:, 1−B:]` 填 `block_output_ids[:, 1:]` 的位置

**pilot idx0 实算例**（spans：vision [15,360)、prompt [0,379)、rollout [379,635)，
T=635，L=256）：
- 合法 anchor 位 p ∈ [379, 633]
- 取 j=10 → p=389：ctx = ctx[:, 0:389, :]（含全部 vision 特征位 15-359 与
  prompt 文本位，以及 rollout 前 10 个 token 的特征 379-388）；
  块占位 389…404；labels = ids[390…404]（rollout 偏移 11…25），权重
  w_m = exp(−(m−1)/7)，m=1…15
- 边界例 j=254 → p=633：块占位 633…648，labels = ids[634] 一个（m=1），
  m=2…15 全 −100（T=635 越界）
- off-by-one 哨兵：p=T−1=634 **非法**（无 label）；c=p 的 ctx 键必须被 mask
  （单测断言项）

### 1e. loss 与训练配置

- 块内 CE，位置权重 **w_m = exp(−(m−1)/7)**（DFlash Eq.4，block16→γ=7；
  γ 随 block size 的对照 10→5、8→4 一并入 config 注释）；m 从 1 起
  （anchor 槽无 loss）；padding/越界位权重 0；按 Σw 归一
- `TrainConfig` dataclass（train_drafter.py）：
  block_size=16, gamma=7.0, alpha=1.5, max_anchors=512,
  lr=6e-4, schedule="cosine", warmup_ratio=0.04, epochs=6,
  grad_clip=1.0, weight_decay=0.01（论文未述，AdamW 惯例值，标注待敏感性检查）,
  optimizer="adamw", dtype=bf16, seed=42,
  batch_seqs=4（等效噪声 token ~1.4 万/步，见 §1g）,
  grad_accum=1, save_best_on="val_weighted_ce", save_every_steps=2000
- 词表：lm_head 物理 152064，loss 前不 mask phantom 行（label 恒 <151936，
  CE 对未命中行无梯度贡献偏置；推理侧沿用 argmax_masked）

### 1f. off-policy 监控埋点（对应 w2_design_todos.md §1）

数据侧：全量 manifest 每条 `n_match/n_pos`（精确）+ `mismatch_head16` 位置
（>16 处截断，覆盖率一并报告）。训练循环每次评估输出：
- `offpolicy_share_exact` = 1 − Σn_match/Σn_pos（epoch 累计，预期 ~4%）
- 分段（基于 head16，近似）：rollout 相对位置十分位 × 长度桶 {<50,50-100,
  100-200,≥200} × token 类型（实词/子词延续/标点/数字，tokenizer 判类）
- 实现：`ctx_dataset` 在样本元数据中携带 (n_match, n_pos, mismatch_pos 列表)，
  训练侧只聚合不重算

### 1g. 显存与吞吐预估（A100 80G 单卡）

| 项 | 估算 |
|---|---|
| drafter 参数 bf16 | 2.46 G（1.23B） |
| 梯度 bf16 | 2.46 G |
| AdamW m+v fp32 | 9.84 G |
| fp32 master 参数 | 4.92 G（如用 bf16-native AdamW 可省） |
| 冻结 embed+lm_head bf16 | 2.18 G |
| ctx 特征 batch（4 序列 × T̄525 × 5×3584 × bf16） | 0.15 G |
| 激活（grad ckpt，逐层重算；峰值≈单层 (T+KB)×3584×若干 + lm_head logits (KB̄3456×152064×2B×4) | ~6-8 G（lm_head 输出是大头，分块计算 CE 可再降） |
| **合计** | **~28-30 G**（80G 冗余充足；batch_seqs 可升到 8） |

吞吐：等效训练 FLOPs（α=1.5）≈ 5.4e18；A100 30% MFU → **~16 GPU·h/次**
（区间 8-27 见 survey §B5c，anchor 数锁定后重估收窄）。
batch 按噪声 token 计：batch_seqs=4 × K̄216 × 16 ≈ 13.8K noise tok/步，
每 epoch ≈ 8.8K 步。

## 2. 代码结构（§2 交付，docstring 注明本文档节号）

- `data/ctx_dataset.py`：`CtxShardDataset`（分片惰性加载 + LRU 单片缓存，
  按 manifest 索引→(shard, key)）、`sample_anchors`（§1b）、
  `pack_blocks`（§1b 布局 + §1d 对齐 + labels/weights）、
  `build_block_mask`（§1c 双路径：`flex_mask_mod` 工厂 + `additive_4d`）、
  collate（ctx/noise 双段 padding）
- `train/train_drafter.py`：`TrainConfig`（§1e）、模型组装（`external/dflash`
  的 `DFlashDraftModel` 原样复用 + Qwen2.5 规格 Qwen3Config 适配 +
  冻结 embed/lm_head 从 base 权重加载）、循环（bf16 autocast +
  grad ckpt + clip + cosine）、评估钩子（weighted CE + off-policy 聚合 §1f）、
  checkpoint 只存 drafter `state_dict`（预期 ~2.5G/档，best + 每 2000 步）
- `pbs/w2_smoke_train.pbs`：模板（哨兵 JSON + 邮件指令沿用 W1 写法），
  **不提交**，随 smoke 授权再用

## 3. 风险与开放项

1. `Qwen3Config` 承载 Qwen2.5 规格：Qwen3 有 q_norm/k_norm（Qwen2.5 无）——
   drafter 是从零训练的独立模型，保留 Qwen3 结构即可（不是复刻 target 层），
   仅 hidden/heads/kv/intermediate/vocab 取 Qwen2.5-VL 值；`layer_types`
   显式全 full_attention，避免 sliding window 分支
2. attention_bias：Qwen2.5 为 True（QKV 有 bias）、Qwen3 默认 False——
   同上理由取 False（drafter 自身结构选择，与 target 无耦合），记入 config
3. γ=7 取自论文 block16；若 W3 消融 block size 需按对照表换 γ
4. weight_decay 论文未述（0.01 惯例值，smoke 后敏感性检查）
5. FlexAttention 与 sdpa 的数值一致性在 GPU 上才可最终确认（CPU 先证 mask
   语义等价）
