# 老线基线固化报告 — Linked Medusa / 动态树

- 执行日期：2026-08-12
- 目标：老线数月后可原样重跑计时，基线定义可复现
- 原则：只读现有产物，未删除任何东西（唯一移动项为第 0 步的违规 tar.gz）

---

## 0. 违规项处理（已完成）

- `~/medusa-qwen/fixed_data_code_pack_local_20260811_203959.tar.gz`（130KB）：
  先比对 sha256（`b8ed0933…`）确认与 scratch 副本字节一致后删除原件，
  原位置建软链 → `/scratch/li96/mz9869/external/mage_pack/` 下的只读副本。
- `.gitignore` 新建，含 `*.tar.gz`（另加 `__pycache__/`、`.pytest_cache/`、`*.pyc` 常规排除）。

## 1a. 活资产清单（须保持可运行）

| 资产 | 路径 | 大小 | 文件 atime | 距 100 天清除线 |
|---|---|---|---|---|
| 冠军 checkpoint（static 与 dyn 共用 C1） | `/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/ckpt_best.pt` | 21.2 GB | 2026-08-09 | **~97 天**（至 ~11-17） |
| B2 对照 checkpoint（final_eval/OOD G3 的 b2_* 配置用） | `/scratch/li96/mz9869/medusa_outputs/linked_medusa_5head_b2/ckpt_best.pt` | 14.0 GB | 2026-08-08 | ~96 天 |
| base_lm_head（推理装配用） | `/scratch/li96/mz9869/medusa_assets/base_lm_head.safetensors` | 1.1 GB | 见前次报告 | 前次已刷新 |
| 34,999 条自蒸馏缓存 | `/scratch/li96/mz9869/cached_data/llava_general_35k/`（`0.pt`–`34998.pt` + `manifest.json`，共 35,000 个文件） | ~60 GB 量级 | **2026-07-13**（训练文件） | **~70 天**（至 ~10-21，全部资产中最近） |
| 树解码推理入口 | `~/medusa-qwen/decode/tree.py`、`decode/common.py` | — | home，无清除风险 | 已入 git |
| 评估 harness | `~/medusa-qwen/scripts/`（eval_acceptance_tree.py、recompute_all.py、clean_greedy_speed300.py、tier_c_interleaved_speed.py、verify_cache_convention.py 等）+ `pbs/*.pbs`（e2e_speed_and_tier3.pbs、dynamic_sweep_300.pbs、ood_mmvet_218.pbs、v2_rebaseline_*.pbs） | — | home | 已入 git |
| 基座模型 Qwen2.5-VL-7B（隐含依赖） | scratch HF hub 缓存（见 mage_pack_inventory.md §2） | 16.6 GB | 2026-08-09 | ~97 天 |

**60 天内到期项：无。** 最紧的是 34,999 缓存（~70 天）。注意本项目 scratch 清除按 atime 计，缓存若两个月内不复跑训练/评估将进入警戒区；建议在 ~10 月上旬前有一次实际读取（或加入 touch 例行）。

## 1b. 基线元数据清单（已归档，见 §3）

| 项 | 路径 | 说明 |
|---|---|---|
| 300-prompt 嵌套 manifest（seed=43） | `eval_manifests/manifest_300.json` | 315 KB |
| 旧 100 回归门 | `eval_manifests/manifest_old100_gate.json` | 字节级回归门标记 |
| 去重记录 | `eval_manifests/manifest_300_dedup_report.json`、`manifest_300_question_collision_freq.json`、`cached_data/llava_general_35k/manifest.json`、`medusa_outputs/cache_source_audit.json/.png` | 双键去重谱系 |
| O0 阴性对照 | `eval_manifests/o0_negative_control_record.json` | |
| MM-Vet 清单 | `ood_eval/mmvet/manifest_mmvet_218.json` + summary（+ 被取代的 100 版及 SUPERSEDED 标记） | |
| 冠军配置 | `medusa_outputs/linked_medusa_c1_eagle/config.json` | hidden 3584、5 heads、cache 指向 llava_general_35k |
| 最终评估结果 | c1_eagle 下 `e2e_speed_300/`（含 e2e_speed_summary.json）、`dynamic_sweep_300/`、`final_eval/`、`ood_mmvet_218/`、`v2_rebaseline/`（含 ARCHIVE_POLICY_V2_ANCHOR.json）、`dynamic_equiv_gates/`；5head_b2 下 `final_eval/`、`c1_g1_widegate.json` | |
| 计时日志 | `/scratch/li96/mz9869/logs/`（71 个 PBS `.OU`，含 e2e 速度作业 175598529、v2 rebaseline 作业 175680071/72）+ `medusa_outputs/*.log`（6 月诊断计时） | |

### ⚠ 口径发现（重要，写入了 tag 附注）

CLAUDE.md 锁定的"static 速度冠军 1.689×"在 [docs/c1_eagle_conditioning.md:59](../docs/c1_eagle_conditioning.md) 与 :176 被标注为**"已废弃口径"**（100-scale segmented，勿与 e2e 混排）。当前 v2 锚定体系（`ARCHIVE_POLICY_V2_ANCHOR.json`，2026-08-07 生效）下的 e2e paired 数值为：static_c1_d3 **1.705×**、速度冠军实为 **dyn_k8_n24 1.732×**（`e2e_speed_300/e2e_speed_summary.json` verdict 字段）。σ 冠军 dyn_k8_n32 σ=2.841 与 Δ+0.101±0.007（对 static_c1_6432 锚 σ=2.7388 的 σ 差）与锁定值一致。**tag 按你指定的数字原样书写，并附注了口径出处**；CLAUDE.md 的 baseline 节是否改用 v2 口径数值，由你判定。

## 1c. 可冷藏项清单（只列不动，处置为【建议】）

| 项 | 大小 | 说明 |
|---|---|---|
| `linked_medusa_5head_b1/` | 99 G | 非冠军线全量输出 |
| `linked_medusa_phaseA_general/` | 59 G | 早期 phase A |
| `linked_medusa_v1_full/` | 59 G | v1 全量 |
| `b2_smoke/`、`b1_smoke/`、`c1_smoke/`、`c1_smoke_nocond/` | 27+20+20+20 G | smoke 输出 |
| `linked_medusa_v1/`、`phaseA_smoke/` | 12+12 G | |
| c1_eagle 非冠军 ckpt：`ckpt_final.pt`、`ckpt_step{6000,7000,8000}.pt` | 4×21.2 G ≈ 85 G | 冠军是 ckpt_best.pt |
| 5head_b2 中间 ckpt：`ckpt_step{200..800}.pt`、`ckpt_final.pt` | 5×14 G = 70 G | b2 保留 ckpt_best.pt 即可 |
| `gpt56_round4–7_release/` | ~200 M | 已交付的 release 包快照 |

冷藏候选合计约 **470–500 GB**（scratch 总用量 4.15 T 的 ~12%）。
【建议】非冠军 ckpt（155G）与 smoke 目录（87G）删除收益最大、重建成本最低（smoke 可随时重跑）；
v1/phaseA 线（~130G）若论文无回溯需求也可删；release 包很小，建议保留。**最终由你判定，本次未动任何文件。**

## 2. 归档操作记录

- 归档文件：`/scratch/li96/mz9869/archives/linked_medusa_baseline_20260812.tar.gz`
- 大小：8,492,550 B（≈8.5 MB；打包前估算 ~65 MB 未压缩，远低于 2 GB 阈值，未触发中断条款）
- 内含 239 个文件（§1b 全部条目 + 完整 PBS 日志目录）
- sha256：`6924f2c629429077d89941d0546bf3b4c032c0bd7c48e78b66a869909a5ebc5e`
- 内容清单：同目录 `linked_medusa_baseline_20260812.manifest.txt`（含 sha256、逐文件 tar 列表）
- **请下载一份到本地**（scratch 100 天规则同样适用于归档本身）：
  `scp gadi:/scratch/li96/mz9869/archives/linked_medusa_baseline_20260812.tar.gz .`

## 3. git 固化记录

- **发现：`~/medusa-qwen` 原本不是 git 仓库**（home 下唯一的 .git 在 `medusa-qwen-archive-qwen3vl/`，属旧归档项目）。为完成 tag，已在项目内 `git init -b main` 并提交当前全部代码与文档（155 文件，commit `e01f6ec`，纯新建仓库、可整体撤销）。此举与"确认工作区干净"的前提不符，故在此显著报告。
- annotated tag：**`linked-medusa-final`** → `e01f6ec`。message 含三项冠军数值（static 1.689×；dyn_k8_n32 σ=2.841，Δ+0.101±0.007；MM-Vet R≈0.433±0.039）+ §1b 的口径附注 + ckpt/缓存/归档路径。
- 无任何分支操作、无 push（无远端）。
- 提醒：git 提交人信息是自动生成的（`mz9869@gadi-login-07`），如需规范可 `git config` 后 `--amend --reset-author`。

## 4. 复现入口速查（数月后重跑计时用）

1. 代码：`git checkout linked-medusa-final`
2. 环境：`module load python3/3.11.0 cuda/12.3.2` + `source ~/medusa-env/bin/activate`（transformers==5.3.0, torch==2.5.1+cu121，见 requirements.lock）
3. 权重：§1a 的 c1 `ckpt_best.pt`（+b2 对照）+ base_lm_head + 基座模型
4. 数据：`manifest_300.json`（in-domain）/ `manifest_mmvet_218.json`（OOD, max_pixels=501760, V100）
5. 入口：`pbs/e2e_speed_and_tier3.pbs`、`pbs/dynamic_sweep_300.pbs`、`pbs/ood_mmvet_218.pbs`、`pbs/v2_rebaseline_*.pbs`
6. 口径定义：归档内 `v2_rebaseline/ARCHIVE_POLICY_V2_ANCHOR.json` 与 `e2e_speed_summary.json`
