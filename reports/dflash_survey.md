# DFlash 官方仓库调研（D0 收尾项）

- 日期：2026-08-12；调研人：Claude（Fable 5），全程未训练、未下载权重
- 论文：arXiv:2602.06036（Chen, Liang, Liu — DFlash: Block Diffusion for Flash Speculative Decoding, ICML 2026）
- 官方仓库：https://github.com/z-lab/dflash
- 本地 clone：`~/medusa-qwen/external/dflash/`，commit `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756`（2026-05-10 "update model list"）

## B1/B2. 仓库清点

**总体结论：这是一个纯推理仓库（代码仅 96K、1492 行），训练代码未开源**
（README 原文 "We will also open-source the training recipe soon"）。B2c 的全部训练细节
（anchor 采样、块内 mask、loss decay）在仓库中不存在，必须从论文自行复现。

```
dflash/
├── README.md            # 安装/快速上手/评测命令/模型列表
├── pyproject.toml       # 依赖声明（四种后端 extras）
└── dflash/
    ├── __init__.py      # 导出 load/load_draft/stream_generate
    ├── model.py         # ★ 核心：DFlashDraftModel + dflash_generate 推理循环（366 行）
    ├── model_mlx.py     # Apple Silicon MLX 版（gemma 用，与我们无关）
    └── benchmark.py     # 评测入口（gsm8k/math500/humaneval/mbpp/mt-bench）
```

- 训练入口：**无**
- 推理/评测入口：`python -m dflash.benchmark --backend transformers|vllm|sglang|mlx`
- drafter 模型定义：`dflash/model.py:302` `DFlashDraftModel`
- 配置文件：无独立配置文件；全部参数在 HF checkpoint 的 `config.json`
  （Qwen3Config + 附加字段 `num_target_layers`、`block_size`、`dflash_config{target_layer_ids, mask_token_id}`）

### B2a. drafter 架构参数在哪配置

`DFlashDraftModel.__init__`（model.py:306-321）从 config 读取：
- 层数 = `config.num_hidden_layers`（drafter 自己的层数，论文 5）
- block size = `config.block_size`（README 例子 16；b16 模型名后缀即 block=16）
- 隐藏维度 = `config.hidden_size`，**直接沿用 target 的 hidden**（Qwen2.5-VL-7B → 3584）
- 抽取层 = `dflash_config.target_layer_ids`，缺省用公式 `build_target_layer_ids`（model.py:27）：
  L_target=28、L_draft=5 时为 **[1, 7, 13, 19, 25]**（start=1, end=L-3, 均匀取整）

### B2b. target feature 抽取方式

- **既不是 hook 也不改 modeling 代码**：直接 `target(..., output_hidden_states=True)`，
  从返回的 `hidden_states` 元组按 `layer_id + 1`（跳过 embedding 层输出）取 5 层
  （model.py:39-45 `extract_context_feature`）
- 拼接投影：5 层沿最后一维 cat（5×3584=17920）→ **单个共享** `fc: Linear(17920→3584, bias=False)`
  → RMSNorm（`hidden_norm`），得到一条融合 context 序列（model.py:334）
- KV injection 接线：融合后的 `target_hidden` 传入**每个** drafter 层，各层用**自己的**
  `k_proj/v_proj` 把它投成该层的 K/V 条目，与 noise block 自身的 K/V 沿序列维拼接
  （model.py:226-231：`k = cat([k_ctx, k_noise])`）；Q 只来自 noise block
- 对照论文验证：5 层均匀采样 ✓；"每层投影为 drafter 各层的 K/V 条目" ✓，但注意
  实现是 **先单 fc 融合 5 层、再由各层 k/v_proj 投影**，不是每个 target 层独立映射到
  对应 drafter 层——所有 drafter 层看到的是同一条融合特征
- 块内注意力非因果（`is_causal=False`，model.py:194），无显式 mask 传入 → noise 块内全互看 + 看全部 ctx

### B2c. 训练数据格式

**仓库无训练代码，全部缺失**：无离线/在线 feature 管线、无 anchor 采样、无块内 mask、
无 loss decay。唯一间接证据：推理时 noise embedding 用 `target.model.embed_tokens(mask_token_id)`
（model.py:111），说明训练时块内未知位置也应是 mask token 的 target embedding。
风险标注：训练配方需从论文复原，官方 "coming soon"——建议 watch 仓库，出了官方 recipe 立即对表。

### B2d. 推理循环（model.py:62-169 `dflash_generate`）

- **纯自定义循环，与 HF `generate` 零集成**（不 patch、不用 GenerationMixin）；bs=1 硬编码
- 流程：target prefill（`logits_to_keep=1` + `output_hidden_states`）→ 取首 token 为 anchor →
  循环：drafter 单次 forward 出整块（block-1 个 draft）→ target 对整块 verify →
  `cumprod` 匹配得 acceptance_length → **bonus token 锚点机制**：`posterior[acceptance_length]`
  （target 在最后接受位置的输出）直接写入 `output_ids[start+acc+1]`，成为下一块的 anchor
  （块首 token 恒为已验证 token，drafter 只填其余 block-1 位）
- KV cache 管理：target 与 draft 各一个 `DynamicCache`，每轮 `crop(start)` 回滚到已接受长度；
  draft cache 里存的是 [ctx K/V + noise K/V] 拼接后的条目，crop 后仅保留已接受位置的 ctx 条目
- verify batch 形态：单序列整块（1×block_size）一次 forward，无树、无多候选——
  与旧线动态树的多分支 verify 完全不同（计时锚点不可比，printCLAUDE.md 已有此约定）

## B3. 依赖与环境核对

### B3a. 版本对比（对照 requirements.lock）

| 依赖 | 仓库声明 | 我们的环境 | 判定 |
|---|---|---|---|
| torch | 未 pin（transformers extra 里裸 "torch"） | 2.5.1+cu121 | ✅ 无冲突 |
| transformers | **==4.57.1**（transformers extra） | 5.3.0 | ⚠️ 声明冲突，但见下：实测兼容 |
| flash-attn | 可选（未装则回退 sdpa 并警告） | 未装 | ✅ 可选，A100 训练建议装 |
| triton | 不依赖 | — | ✅ |
| datasets/rich/loguru 等 | 裸声明 | 已有/易装 | ✅ |

**实测（本机 login 节点，transformers 5.3.0 + torch 2.5.1）**：`dflash.model` 全部
import 成功（Qwen3 modeling 的 10 个符号、`DynamicCache`（含 `crop`/`get_seq_length`）、
`DFlashDraftModel` 类本体），`build_target_layer_ids(28,5)=[1,7,13,19,25]` 验证通过。
4.57.1 的 pin 属保守锁定，**无需降级**；风险面收窄为运行期行为差异（后续 smoke 再验）。

### B3b. FlexAttention

**仓库完全没有使用 FlexAttention**（也没有 triton、torch.compile）。注意力通过
`ALL_ATTENTION_FUNCTIONS[config._attn_implementation]` 分发（model.py:239-241），
支持 eager / sdpa / flash_attention_2；benchmark.py:185-193 优先 flash_attention_2，
未安装则**自动回退 sdpa**。论文训练若用 FlexAttention 做块状 mask，属于我们自研训练
代码的选型问题：torch 2.5.1 已含 FlexAttention（2.5 引入），A100（sm80）可用；
但我们的训练 mask 也可以用普通 4D additive mask + sdpa 实现，不强依赖。

### B3c. V100（sm70）硬件兼容性专项 —— OOD gate 锚定判定

- flash-attn 2：**要求 sm80+，V100 不可用**
- FlexAttention / triton / torch.compile：sm70 官方不支持（triton 面向 sm80+ 优化，
  torch.compile 在 sm70 上无保障）——**均不可用**
- 推理回退路径：**可行**。DFlash 推理只需 eager/sdpa 之一（B3b），sdpa 在 V100 完全可用；
  drafter 与 target 用同一 `attn_implementation`，无任何 kernel 硬依赖
- 计时公平性：paired 协议（greedy 与 spec 同硬件、同 attn 实现、同进程交替）下，
  比值型指标（speedup、σ→realization）**内部公平不受影响**；受影响的是绝对 tok/s
  （sdpa 慢于 flash-attn），以及与 A100 数字的跨硬件混排（本来就禁止）
- **结论：OOD gate 可以继续锚定 V100**，条件是预注册中写死
  `attn_implementation="sdpa"`（greedy 与 spec 两侧一致），并沿用旧线"计时锚点重新
  预注册"的约定。max_pixels=501760 的 V100 显存约束与旧线相同，不新增风险。

## B4. Qwen2.5-VL 迁移改动清单（只列清单，不写代码）

### B4a. feature 抽取接线 + vision token 设计决策

改动点（难度：低）：
1. target 换成 `Qwen2_5_VLForConditionalGeneration`；`output_hidden_states=True` 返回的
   就是 decoder 逐层 hidden（vision 已作为占位 token 融进序列），`extract_context_feature`
   原样可用；抽取层 [1,7,13,19,25]
2. prefill 调用需带 `pixel_values/image_grid_thw`（复用 `decode/common.make_image_inputs`）；
   `logits_to_keep=1` 在 5.3.0 的 Qwen2.5-VL forward 上需 smoke 验证
3. drafter 的 noise embedding 用 `target.model.embed_tokens`（纯文本 token，不涉 vision）

**设计决策点：vision token 是否进 context feature**（论文是纯文本 LLM，无此问题）：
- 方案 A（vision 进 ctx，完整序列）：忠实于"context feature = target 全上下文"；drafter 可
  直接看到视觉信息，图文相关 rollout 的接受率上限更高。代价：ctx 平均长 +~330（实测
  prompt+vision 均长 381 vs 纯文本 prompt ~50），离线缓存 ~660 GB（见 B5a），draft KV 更大
- 方案 B（只取文本位，vision 位剔除）：缓存降到 ~240 GB、推理 ctx K/V 减半以上；代价：
  drafter 对图像内容盲，接受率在视觉描述段预计显著下降（我们数据全部是看图任务），
  且位置对齐簿记复杂化
- **倾向方案 A**（数据 100% 是图像任务，接受率校准是核心研究问题，不应先天砍掉视觉通道；
  660 GB 在 scratch 余量内）。A/B 可作为后续单变量消融，方案待确认

### B4b. 共享 embedding / LM head

- 查证 config：Qwen2.5-VL-7B `tie_word_embeddings=false`，**embedding 与 lm_head 是两套权重**
- DFlash 接法：drafter **不含**自己的 embed/lm_head（推理时直接调 `target.model.embed_tokens`
  与 `target.lm_head`，model.py:111-112），迁移后同样直接引用，两套权重照常各用各的，
  无需任何 tie 假设
- 参数量影响：drafter 可训练参数 ≈ **1.23B**（5 层 Qwen2.5-7B 规格：每层 attn 29.4M
  [GQA 28q/4kv] + MLP 203.7M[intermediate 18944] ≈ 233M，×5 + fc 17920×3584=64.2M）；
  embed/lm_head（各 152064×3584≈0.545B×2）不计入训练参数。离线缓存路线训练时需在显存
  载入这两块（bf16 各 1.09GB；lm_head 已有本地资产 base_lm_head.safetensors 可复用）

### B4c. M-RoPE 与 drafter 位置编码

- 仓库（Qwen3 纯文本）做法：drafter 内建 `Qwen3RotaryEmbedding` 一维 RoPE，直接吃全局
  `position_ids`（0..T-1 连续整数），ctx K 与 noise Q/K 同一套（model.py:176-182：
  q 只取 cos/sin 的尾部 q_len 段）
- Qwen2.5-VL target 用 M-RoPE（mrope_section [16,24,24]，3D position_ids）——但这只影响
  **target 内部**；drafter 是从零训练的独立小模型，**自建一维 RoPE 即可**（文本 token 在
  M-RoPE 里三分量本就相同；vision token 在 drafter 的一维平铺位置下由训练自行适应）
- 结论：**沿用仓库做法（drafter 一维 RoPE、扁平位置）**，不引 M-RoPE 进 drafter；
  唯一注意点是 ctx 与 noise 的 position_ids 必须与训练时一致（扁平序列位）
- 难度：低（不改，仅换 RotaryEmbedding 的 config 来源为 Qwen2.5 规格）

### B4d. 34,999 缓存可否复用

**判定：context feature 不可复用，必须重新抽取。** 实测证据：
- 现存格式（`gen_cache_rollout.py:14`）：`{hidden: (L≤256, 3584) fp16, tokens: (L,)}`，
  hidden 是**逐 emit 步的最终层（第 28 层出口）**，TARGET convention
- DFlash 需要：**[1,7,13,19,25] 五个中间层**、覆盖**完整上下文（vision+prompt+rollout）**
  ——现缓存既无中间层、也无 prompt/vision 段
- **可复用部分**：`tokens`（34,999 条 greedy rollout 轨迹本体）与 `rollout_prompts.json`
  + images（去重谱系不变）。重抽只是对固定 (prompt, rollout) 重跑 target forward 记录
  中间层，**数据谱系与三层评估体系完全不动**

## B5. 存储与算力预算（实测数据推算）

### B5a. 离线 5 层 feature 缓存总量

实测输入（本日登录节点测量）：
- 条数：34,999（manifest：n_cached=34999, complete=true）
- rollout 长度（400 条抽样）：均值 144.4，中位 121，p90 256（截断），min 8
- prompt+vision 总长（200 条抽样，processor 实测，max_pixels=501760）：
  均值 381.0，中位 380，p10 264，p90 431，max 565
- 完整序列均长 ≈ 381 + 144 ≈ **525 token**

**方案 A（vision 进 ctx）**：34,999 × 525 × 5 层 × 3584 × 2B(bf16)
= **≈ 658 GB ≈ 0.60 TiB**
**方案 B（纯文本 ctx，prompt 文本≈50）**：34,999 × 194 × 5 × 3584 × 2B ≈ **243 GB**

### B5b. 三条路线对比

当前水位（本日 lquota）：scratch **4.17/10.00 TiB**（余 5.8T）；
inode **964.8K/1018K（余仅 ~53K，硬限 1068.9K）**——这是主要约束。

| 路线 | 存储 | 时间开销 | 评价 |
|---|---|---|---|
| ① 全量离线缓存 | 658 GB（方案 A），**必须打包**：按 ~256 条/分片打成 ~137 个 safetensors 分片（每片 ~4.8G），inode 消耗可忽略；>50G 写入，写前写后跑 lquota | 一次性抽取 ~2-4 A100·h；训练时零 target 开销 | **推荐** |
| ② 在线抽取 | 0 | 每 step 增加 target forward（28 层+ViT vs drafter 训练 ≈5 层×3）；FLOPs 占比约 +15-20%（anchor 批内共享 ctx 时），但显存 +~16GB（bf16 target）、数据管道需带图像预处理（CPU 端历史瓶颈） | 可行但工程复杂 |
| ③ 混合（只缓存 prompt 段） | ~477 GB（缓存 prompt+vision 段 381×5 层，rollout 段在线） | 在线段仍需 target forward | 两头不讨好，不推荐 |
| ①′ 顺带说明 | fp16 与 bf16 同字节数，无廉价降精度空间（fp8 缓存属未验证口径，不预设） | | |

**推荐路线 ①**，理由：(1) target 冻结 + rollout 固定 greedy → 特征**确定性可复现**，
契合本项目预注册/exactness 纪律，训练与抽取解耦后 bug 面最小；(2) 训练时不占 target
显存，batch/MFU 更高；(3) 658GB 对 5.8T 余量无压力，打包后 inode 影响为零；
(4) 在线路线的图像预处理 CPU 开销在 34,999×6 epoch 下重复浪费 6 次。
风险：scratch 100 天 atime（训练期高频访问，不触发；空窗期参照缓存 touch 纪律）。

### B5c. 训练算力与 SU

**SU 费率实证校准（2026-08-12，从 PBS 记账反推）**：
- dgxa100（16 核/GPU 形态）：job 173349156 = 549.82 SU / 7h38m = **72.0 SU/GPU·h**；
  job 173571072 = 93.98 SU / 1h18m = 72.0，两作业精确吻合（= 16 核 × 4.5 SU/核·h）
- gpuvolta V100（12 核/GPU）：job 171434109[0] = 200.32 SU / 5h32m ≈ **36.2 SU/GPU·h**
- 早先按 "4.5 SU/GPU·h" 的估算**低了 16 倍**，以下全部按实测 72 重算

估算假设（6 epochs、5 层 drafter、block=16；anchor 同序列批内共享 ctx 投影）：
- **anchor 数修订**：anchor = min(512, α×有效位置数)，α∈[1,2]；按 rollout 均长 144
  估计实际 **150–300 anchor/序列**（论文的 512 是长文本 LLM 语境，我们的 rollout
  均长仅 144，512 会大量重复采样）。**最终值 W2 代码审查时锁定**
- 每 anchor fwd+bwd ≈ 6 × 1.23e9 × 16 ≈ 1.18e11 FLOP；每序列每 epoch
  150–300 anchor ≈ 1.8e13–3.6e13 FLOP（ctx 侧摊销 +~2%）
- 总量 ≈ 34,999 × 6 epochs → **3.7e18–7.4e18 FLOP**（若保守取满 512：1.30e19）
- A100 bf16 按 25–40% MFU（78–125 TFLOPS 有效）→ **单次训练 8–27 GPU·h**
  （512 anchor 上界：29–46 GPU·h）；feature 抽取另 +2–4 GPU·h
- SU 换算（实测 72 SU/GPU·h）：单次训练 **0.6–2.0 KSU**（512 上界 2.1–3.3 KSU）；
  含 3 次全量重训 + 抽取 + 评估的总预算 → **约 4–8 KSU**（anchor 取满 512 的
  最保守情形 ~13 KSU）
- 当前余额（本日 `nci_account`，2026.q3）：**Avail 137.73 KSU**（Grant 300，Used 161.55）
- **判定：预算仍充裕**（总预算占余额 3–9%），但不再是"忽略不计"量级；
  重训次数需纳入管理，避免无审查的试错训练

## W2 参考资料（训练配方自研前必读）

- **github.com/hikarioyama/dspark-aeon-27b**：社区完整训练 recipe（DFlash backbone
  微调）。官方训练代码未开源背景下目前最完整的第三方实现，W2 动工前通读
- **arXiv:2606.02091（DFlare）**：训练流程描述比 DFlash 原文更细，
  多 anchor 稀疏 mask 的实现参照
- **训练量级风险**：z-lab 官方训练量级为 **800K 样本**，我们只有 35K（约 1/23）。
  D1 gate 阈值预注册时按小数据折扣设定，**不锚论文 τ=6.5**；接受率/σ 的预期
  应以我们自己的训练分布评估为准，论文数字仅作方向参考

## 收尾水位（2026-08-12 lquota）

- scratch：4.17 / 10.00 TiB（limit 10.5T），inode 964,775 / 1,018,000（limit 1,068,900）
- gdata：3.94 / 10.00 GiB，inode 2,281 / 36,000
- 提示：inode 余量 ~53K 是当前最紧约束；D1 一切新产物必须分片打包
