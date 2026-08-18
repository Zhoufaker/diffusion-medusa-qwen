# D2 正式评估报告（三臂 × 三次重复计时）

2026-08-19。协议：reports/d2_eval_protocol.md v1.2（预注册锁定）。
作业：r1=176579877（80.18 SU）/ r2=176595080（79.90）/ r3=176595082（77.16）。
n_scored=295/跑（300 − 5 warmup）；ckpt=drafter_best_final_pinned.pt
（epoch 3，sha256 97b9d413…）；harness commit 与冒烟一致（0 diff 已验）。

## 结论先行

**G1 唯一硬门:通过。** diffusion drafter（block 16）在 per-token paired
加速比上超越旧线速度冠军 dyn_k8_n24,三跑合并 CI 不跨零。

## G0

- **G0a 代码保真（V100 硬门）**：通过——100/100 字节级一致（job 176356106）
- **G0b 计时基准（A100）**：树冠军 per-token 加速比三跑 1.8528/1.8526/1.8509,
  **新基准 ≈1.852×**（V100 时代 1.732×,硬件迁移登记,无数值对照门;
  σ_batch=0.0537 仅登记）

## G1（协议扩跑分支全程记录）

| 跑 | point（Δ=spd_drafter−spd_tree） | CI95 | 单跑判定 |
|---|---|---|---|
| r1 | +0.0115 | [−0.0039, +0.0267] | marginal（跨零）→ 触发扩跑 |
| r2 | +0.0172 | [+0.0028, +0.0318] | 不跨零 |
| r3 | +0.0153 | [+0.0008, +0.0303] | 不跨零 |
| **合并（写死规则：每 prompt 三跑均值的 paired 差,bootstrap 10k,seed=43）** | **+0.01466** | **[+0.00032, +0.02929]** | **PASS** |

- 附注：CI 下界贴近零（+0.0003）——统计上过线,幅度上是"小胜"
  （drafter ≈1.867× vs 树 1.852×,差 ~0.8%）。叙事建议交导师：速度平手偏上
  + σ/τ 与实现简洁性为主要卖点
- 输出侧确定性旁证：τ 与 cross_exact 三跑逐位一致,波动纯在计时侧

## G2 确定性

三跑各 3 条双跑,256 token 全长逐 token 一致——**9/9 通过**。

## τ 两口径（登记,协议脚注纪律）

- τ_accept_only = **1.531**;τ_with_bonus = **2.531**（三跑一致）
- 冒烟倒推线 τ_wb~2.4 → 实测 2.531 略超
- 对照 DFlash 论文 τ=6.5：**必须带数据折扣脚注——官方 800K 样本训练,
  我们 35K（1/23）**;且本档为 5/6 epochs 提前收账档

## 跨臂匹配率与分叉抽检（greedy 等价性声明登记项）

- 逐条完全一致 **205/295（69.5%）**,三跑一致;分叉率 ~0.16%/位置
  （与四臂矩阵 A100 同硬件预期吻合,好于协议保守值一个量级）
- **10 条分叉抽检（全文见报告尾注数据路径）**：全部为实词选择位的
  同义/近义改写（culture→traditions、suggesting→indicating、meals→eating、
  hygiene→personal…）,分叉后语句通顺无质量退化——漂移型确认,
  与"lossless w.r.t. teacher-forcing verification"定义相容

## D3 埋点完整性清点

- **300 文件/跑 × 3 跑全齐**,逐 cycle 记录（slot top-1 概率/accept/
  reject 处 verify argmax）,~83 cycle/prompt
- 数据路径:/scratch/li96/mz9869/dflash_data/d2/{formal,formal_r2,formal_r3}/d3/
- D3 分析(块内位置接受率曲线 + per-slot 置信度)与 ep3vs4 校准对比为下一块

## off-policy 附属实验（预注册,job 176578425）

on 层（95.8%）drafter slot-1 命中 verify-argmax **69.98%** vs off 层（4.2%）
**31.96%**——方向性预期成立;接受率损失上界 = 4.2%×38pp ≈ **1.60%**。

## SU 台账

| 项 | SU |
|---|---|
| G0a | 10.13 |
| 冒烟（含失败首跑） | 13.50 |
| 门4+off-policy 合并单 | 9.88 |
| 正式三跑 | 237.24 |
| **D2 线合计** | **270.75**（扩跑预授权 ≤250 内:扩跑部分 157.06 ✓） |

逐 prompt 原始 JSON:/scratch/li96/mz9869/dflash_data/d2/*/per_prompt/。

## 后续（待授权/裁决）

1. G1"小胜"幅度的论文叙事定位（与导师）
2. D3 深挖 + ep3vs4 校准对比(零训练成本,harness 复用)
3. D4(MM-Vet OOD,V100+sdpa)协议起草
