# W1 pilot 报告 — 5 层 ctx feature 抽取（500 条）

- Job：176143796.gadi-pbs（dgxa100 1×A100），exit 0，walltime 3m48s（申请 2h）
- 实际费用：**4.56 SU**；代码 commit `912c7988`（manifest 内绑定）
- 产物：`/scratch/li96/mz9869/dflash_data/ctx_cache_35k_pilot/`
  shard_00000（4.77GB，256 条，partial=False）+ shard_00001（4.63GB，244 条，partial=True）
  + ctx_manifest_pilot.json（345KB）
- **结论先行：性能/格式/预算全部达标；exactness 双门槛未过（系统性、非近平局），
  按预注册纪律停止，全量抽取不启动，等你裁决。**

## 1. 吞吐与资源

| 指标 | 实测 | 备注 |
|---|---|---|
| 吞吐 | **216.7 条/min；1,894 tok/s** | 500 条净算约 2.3 min |
| 峰值显存 | **16.2 GiB**（torch allocated）/ 17.7GB（PBS） | 200GB 申请大幅冗余；**V100 32GB 也放得下** |
| GPU 利用率 | 35%（PBS 口径，含 1 分钟模型加载） | 纯抽取段更高 |
| 失败样本 | **0 / 500** | |

## 2. 单片大小与全量外推

- 实测 18.4 MB/条（bf16、均长 T≈520），单满片（256 条）≈ **4.77GB**
- 全量外推：**~640GB / 137 片**，比 survey 估算 658GB 低 3%（长度采样噪声内）
- 全量 walltime：34,999 条 ÷ 216.7/min ≈ 161 min + 加载 ≈ **~2.8h 单卡**
- 全量 SU（72.0 SU/GPU·h 实测费率）：**~200–220 SU**
- inode：137 片 + manifest ≈ 140 个文件，可忽略

## 3. 格式与谱系验证（抽查通过）

- shard0 键数 768 = 256×3（ctx/ids/spans）✓；抽 3 条：ctx (5,T,3584) bf16、
  ids (T,)、spans 与 manifest 逐项一致 ✓
- spans 示例 idx0：vision [15,360)、prompt [0,379)、rollout [379,635)——
  vision/prompt/rollout 三段边界正常
- PREPROCESS_SPEC 冻结确认：`Qwen2VLImageProcessorFast`，
  size={shortest_edge 3136, longest_edge 12845056}（默认像素，无 501760 cap）✓

## 4. ⚠ exactness 审计：双门槛未过

| 指标 | 实测 | 门槛 | 判定 |
|---|---|---|---|
| 样本级完全一致率 | **13.6%**（68/500） | ≥99.5% | ❌ |
| 位置级一致率 | **95.93%**（mismatch 2,873/70,616） | ≥99.9% | ❌ |

### mismatch 分布（预注册要求的分桶）

- **near-tie 占比仅 27.4%**（|logit gap|<0.05）；72.6% 为大 gap
  （中位 0.125，p90 0.516，max 5.09）——**不是 fp16 近平局翻转能解释的量级**
- 按相对位置（rollout 内十分位）：345→210 缓慢递减，**无随深度累积**
  （teacher forcing 下也不应有）；各处均匀分布 → 系统性偏移
- 按样本长度：L<50 的样本 50.4% 完全一致，L≥100 的近乎 0%——与
  "每位置独立 ~4-6% 翻转率"的复合概率一致（0.96^L）
- 逐样本 mismatch 直方图平滑（0 条 68 个样本、1-11 条各 17-46 个），
  非个别坏样本，**几乎每条长样本都有少量翻转**

### 归因排查（已做）

1. **预处理/版本漂移：排除。** transformers 5.3.0 于 2026-03-23 装入 venv，
   6 月 rollout 与本次抽取同环境、同 fast processor、同 `make_image_inputs`
   代码路径；spec 冻结值与旧脚本行为一致
2. **剩余两个候选（无法在事后区分，需对照实验）**：
   a. **硬件**：旧缓存生成于 V100（gpuvolta，2026-06-17），本次 A100——
      fp16 GEMM/sdpa kernel 数值路径不同
   b. **前向形态**：rollout 是增量解码（prefill + 逐 token KV cache），
      本次是整序列一次 teacher forcing——注意力 kernel 与累加顺序不同
   greedy rollout 的每个采出 token 本就常处于 top-1/top-2 边界附近，
   微小 logit 扰动即可翻转 ~4% 位置，与观测吻合

## 5. 处置选项（等你裁决，全量抽取已按纪律搁置）

- **选项 A：2×2 归因实验**（50 条 × {V100, A100} × {增量, 整序列}，<10 SU）。
  分清硬件与前向形态各占多少；若 V100+增量能还原 ~100%，证明管线无 bug、
  纯数值漂移，为选项 B/C 提供依据。新入口参数 + 队列变更 → 必审档，
  我可先发实验脚本与预算
- **选项 B：重定义审计基准后放行全量。** 论据：ctx feature 的本质是
  "target 读这条固定序列时的中间态"，与 rollout 当年怎么采出来的无关，
  (ctx, tokens) 自洽性在 teacher forcing 下天然成立；审计门改测
  "抽取自身的确定性"（同卡同 commit 重跑 500 条逐字节一致）+
  如实记录对 V100 增量基准的 4% 漂移。代价：drafter 学的轨迹里有 ~4% token
  不再是 A100 目标模型的 top-1（轻度 off-policy），对接受率校准的影响
  需在训练分布评估中显式监控
- **选项 C：V100 上跑全量抽取**（显存 16.2GiB 放得下 32GB V100）。
  消除硬件差，只留前向形态差（预计残余漂移大幅缩小但非零）；
  gpuvolta 36.2 SU/GPU·h，吞吐未知（估 2-4× 慢 → ~6-11h，~220-400 SU），
  且与"训练/推理都在 A100"的最终用途错位
- 不建议：重造 rollout（毁数据谱系锁定，且新 rollout 在别的硬件上同样不可复现）

## 6. 水位与状态

- lquota：scratch 4.01 TiB / 10 TiB；**项目 inode 761,275 / 1,018,000
  （余 256.7K）——警戒解除**（其他成员清理了 ~244K）；pilot 产物 +3 文件/9.4GB
- git 干净；本块无新 commit（分析只读）
- 待办（全量提交前）：哨兵文件 + 邮件指令写入 extract 脚本与 PBS 模板
  （pilot 已结束，冻结解除，等裁决后与全量改动一并做）
