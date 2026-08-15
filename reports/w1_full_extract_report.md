# W1 全量抽取报告 + 条件收编执行记录

2026-08-15。Job 176267512（dgxa100 1×A100），exit 0，2h41m，**193.14 SU**。
产物：`/scratch/li96/mz9869/dflash_data/ctx_cache_35k/`。

## 1. 门 1 复验（抽取确定性）

**通过**：全量作业收尾内嵌比对，本次重算的 shard_00000 与 pilot 产物
sha256 **逐字节一致**（`05b8b6b7…`）；与 rerun256 作业（176264874）的独立双跑
一致结果互为印证。代码谱系：`extract_ctx_features.py` 自过审 commit `f8ff4e8`
后零改动（`git diff` 空），运行时 commit `4c71fada` 仅含无关文件。

## 2. 全量统计

| 项 | 值 |
|---|---|
| 成功 / 失败 | **34,999 / 0** |
| 总长 T（vision+prompt+rollout） | mean 527.0，med 519，p10 385，p90 680，max 832 |
| rollout 长 L | mean 142.6，med 121，p10 13，p90 256（截断），max 256 |
| 实际体积 / 片数 | **616 GiB / 137 片**（预估 640GB，−4%；单满片 ~4.6G） |
| 吞吐 | 218.7 条/min，1,921 tok/s，峰值显存 16.18 GiB |
| manifest | ctx_manifest.json（hardware=A100-dgxa100、PREPROCESS_SPEC、逐条 spans/sha256/审计） |
| L<2 样本 | 0（train_manifest 复核 n_filtered_short=0） |

## 3. 门 2 漂移登记（不设通过线）

全量位置级一致率（vs V100 增量基准）＝ **95.863%**（样本级完全一致 14.18%）。
与 pilot 抽样 95.93% 一致。原 99.5%/99.9% exactness 门已废止（归因见下）。

## 4. 四臂矩阵补全（50 条 panel，8,587 个 rollout 位置）

| 比较 | 位置级一致 | 归因实验预测 | 对照 |
|---|---|---|---|
| Arm1 (V100+增量) vs 旧缓存 | 100.000% | — | 管线无罪（既有结论） |
| Arm2 vs Arm1（AR 侧纯硬件） | 逐位翻转 ~0.155% | — | 既有结论 |
| Arm3b (A100 TF panel) vs 旧缓存 | 96.099%（样本级 8.0%） | ~96% | ✓ |
| **Arm4 (V100 TF panel) vs 旧缓存** | **96.169%**（样本级 8.0%） | **~96%（前向形态 ~4%）** | **✓ 3.83% 吻合** |
| **Arm4 vs Arm3b（TF 侧纯硬件）** | **99.790%**（8,569/8,587；样本级 37/50） | **~99.8%（~0.2%）** | **✓ 0.21% 吻合** |

**结论闭环**：漂移 = 前向形态 ~3.8-3.9%/位置（主导，V100/A100 两侧独立复现）
+ 硬件 ~0.16-0.21%/位置（AR/TF 两侧独立复现），四臂交叉自洽，无残余未解释项。

## 5. 条件收编执行记录（门 1 通过 + 失败=0 → 自动生效）

### 5a. llava_general_35k
- tar：`archives/llava_general_35k_legacy.tar`（**35,882,895,360 B ≈ 33.4 GiB**，不压缩）
- 三重验证：条目 35,000/35,000 ✓；随机 100 文件 sha256 全一致（seed=42）✓；
  manifest.json 逐字节一致（cmp）✓
- 散件已删除；原路径留 `README_MOVED.txt`（tar 位置/验证记录/解包命令/谱系说明：
  tokens 已逐条 sha256 绑定进 ctx_manifest，hidden 仅旧线使用）

### 5b. onpolicy_data
- 散件已删除（tar 于 2026-08-13 建立并三重验证，5.29 GB）；
  原路径留 `README_MOVED.txt`（B2 重训场景使用说明 + 解包命令）
- W1 抽取输入依赖已解除（全量完成），CLAUDE.md 数据资产锁定条款兑现

### 5c. 水位对照

| 时点 | scratch 用量 | 项目 inode | 我们名下文件数 |
|---|---|---|---|
| 收编前 | 4.63 TiB | 758,675 | 114,177 |
| tar 后（+33.4G） | 4.66 TiB | 758,676 | — |
| **删除后** | **4.63 TiB** | **692,934（−65.7K）** | **49,648（−64.5K）** |

⚠️ **与"预期 <5K"的出入**：剩余大头为 `cached_data/qwen25vl_long/`
（**46,615 个文件**，未在本次收编授权范围，此前登记为"新线资产"）。
扣除它后我们名下 ≈3.0K，与预期吻合。**其处置待你裁决**——若确属废弃/可收编资产，
同法 tar+删可将名下 inode 降至 ~3K。

## 6. 长期资产 touch 名单更新建议（已写入 CLAUDE.md）

- archives/llava_general_35k_legacy.tar（33.4G，2026-08-15，三重验证通过）
- archives/onpolicy_data_legacy.tar（既有条目，散件已删的说明需同步）
- archives/qwen25vl_long_v1cache.tar（~77G，见附录 A）
- dflash_data/ctx_cache_35k/（616G，训练期高频访问不受 atime 威胁；训练间歇期注意）

## 附录 A：qwen25vl_long 验明正身盘点（2026-08-15，只读）

**结论：此目录是 v1 三头训练缓存（项目初期外部传入），"新线资产"的旧登记有误。**

| 检查项 | 结果 | 判定 |
|---|---|---|
| a. 构成 | 46,614 个 `.pt` + 1 个 manifest.json = 46,615 文件；77G；**mtime 全部 2026-04**（项目初期） | ✓ |
| b. 抽样结构（idx 0/1000/23000/46000/46613） | 全部为 `{hidden: (256, 3584) fp16, tokens: (256,) int64}`——定长 256、TARGET convention 的 v1 缓存 schema（`verify_cache_convention.py` 与 `train/loss.py:22` 的注释即为此缓存所写） | ✓ |
| c. v1 记录交叉验证 | manifest n_cached=46,614 complete=true；**77G 与 [linked_medusa_spec.md:18] "The 77 GB cache is still being transferred to …/qwen25vl_long/" 精确吻合**；`config/linked_medusa_default.yaml:22` 以此路径为 v1 训练 cache_dir | ✓ |
| d. 全仓 grep | 命中全部属旧线（linked_medusa_spec.md ×4、cache_source_audit.py ×4、config yaml ×2、tests/test_cached_dataset.py（cached_data_test 变体）、verify_cache_convention.py、train/loss.py 与 data/cached_dataset.py 的注释/默认路径——均为旧线模块）；**新线入口（ctx_dataset / train_drafter / extract_ctx_features / w1·w2·drift PBS）及其 import 闭包零引用** | ✓ |

三项确认齐 → 条件收编生效：tar `archives/qwen25vl_long_v1cache.tar` + 三重验证
+ 删散件 + README_MOVED（"v1 三头训练缓存，马哥提供，冻结基线
linked-medusa-final 的训练数据；重训 v1 场景使用"）。验证与水位见附录 B。

## 附录 B：qwen25vl_long 收编验证与最终水位（2026-08-15 执行）

- tar：`archives/qwen25vl_long_v1cache.tar`（**81,976,391,680 B ≈ 76.3 GiB**，不压缩）
- 三重验证：条目 **46,615/46,615** ✓；随机 100 文件 sha256 全一致（seed=42）✓；
  manifest.json 逐字节一致（cmp）✓
- 散件已删除；`README_MOVED.txt` 就位（身份勘误 + tar 位置 + 验证记录 + 解包命令）

### 最终水位（三次收编累计）

| 时点 | scratch | 项目 inode | 名下文件数 |
|---|---|---|---|
| 收编前基线（2026-08-15 晨） | 4.63 TiB | 758,675 | 114,177 |
| llava+onpolicy 收编后 | 4.63 TiB | 692,934 | 49,648 |
| **qwen25vl_long 收编后（最终）** | **4.66 TiB** | **646,331** | **3,040** |

- 名下 inode 累计释放 **111,137**（114,177 → 3,040，达成"<5K"目标）
- 项目 inode 余量 371.7K（软限 1,018K），警戒彻底解除
- scratch 用量净变化 +0.03T（三个 tar 合计 115.6G 入 archives，散件 116G 删除）
