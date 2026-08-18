# D1.5 长文本数据固定报告（阶段一：盘点与固定）

2026-08-19。零 GPU、零 qsub。协议权威：mage_pack `longform_dataset_registry.py`
（sha256 `7e736eb0…`，项目副本 external/，原包只读）。
产出：`/scratch/li96/mz9869/dflash_data/longform_fixed_v1/`。

## Phase A：库存盘点

### A1. Registry 协议要点（逐源记录）

| 项 | 协议内容 |
|---|---|
| question 生成 | 记录自带 question/prompt/query/instruction 字段则用之；**DOCCI/DetailCaps 均无 → 全部落到固定模板 `LONGFORM_PROMPT`**（"Describe this image in detail. Include the main objects, scene context, relationships, visual attributes, and any uncertainty."）——rollout prompt 形态即此，原样保真 |
| 固定顺序 | HF 路：`ds.shuffle(seed=42)` 顺序取前 n；JSONL 路：`random.Random(42).shuffle(records)`。seed=42 为协议默认（覆盖项，未动） |
| 样本 schema | {image PIL, question, reference, source, idx, sample_id}；reference 字段链 reference/answer/…/description/text/response，列表取最长 |
| 导出行为 | 逐张 jpg（q=95）+ JSONL——**唯一授权改造点**，已改 webdataset tar + 原始字节（不重编码，[EXPORT-MOD] 注释在案） |

### A2. 直接可取源探测

- **DOCCI**：HF 路**实测不可取**——主分支仅 script-loader（datasets≥3 不支持），
  `refs/convert/parquet` 分支是 3MB 存根（default/train/0000.parquet 单文件，
  API 树列表在案）。**改走官方发布包**（storage.googleapis.com/docci：
  descriptions.jsonlines 10MB + images.tar.gz **7.59GB**，HEAD 实测 <20GB 阈值）
  → registry 认可的 `--data-path` 本地 JSONL 路（与占位源同款准备方式）
- **DetailCaps-4870**：HF parquet 971MB / 4,870 行，直接可取；
  但 schema 与 registry 字段链错位（见 B2 决策点）
- `datasets>=3,<4` 装入 venv（dev-dep 登记，不入 requirements.lock）

### A3. 四个占位源
localized_narratives / stanford_paragraph / vqaonline / vizwiz_lf：
registry 明文要求自备本地 JSONL——**登记"需自备,本轮不做"**。

### A4. detail 源排除
LLaVA-Bench-In-the-Wild 族：**整体排除**。理由：与 in-domain 评估语料同族
（LLaVA 生态评估集，且我们评估 prompt 源于 LLaVA-Instruct）——训练-评估污染。

### A5. 盘点表（可取源实测）

| 源 | 可取条数 | ref token 长（Qwen tokenizer 实测） | 图像分辨率 | 许可证 |
|---|---|---|---|---|
| DOCCI | **14,847**（train+test+qual 全 split） | **mean 139.9**，med 130，p10 84，p90 209，max 563；**≥300 的仅 1.1%** | 中位 2048×1536，max 边 4032 | CC BY 4.0 |
| DetailCaps-4870 | 4,870（**本轮 0 条入库**，见 B2） | GT caption 目测数百 token（未入库未统计） | —（源含 COCO/SAM/LAION 图） | 见数据卡 |

⚠ **长文本属性预警（如实报告，不剔除）**：DOCCI 均长 139.9 远低于
"mean ≥300"预期——它是"密集细节描述"而非超长文本。对 max_new_tokens
裁定（512 vs 384）与"长文本"叙事的影响列入决策点。

## Phase B：固定与打包

### B1. 固定结果（本轮入库 = DOCCI）

- **14,847 条 / 15 片 webdataset**（shard_00000-00014，~1000 对/片，7.2G，
  原始字节未重编码）+ `triplets.jsonl`（14,847 行，含 ref_token_len）+
  `manifest.json`（seed/registry sha256/commit/许可证/去重摘要）
- 抽检 5/5 通过：图像可解码、reference 与源 descriptions 逐字节一致、
  question == 协议模板
- 事故与修复记录（全程 git 在案 `7ac1adb`/`eeefe14`）：首跑被登录节点
  30min CPU 上限杀死 → shard_00014 截断 + triplets 缓冲丢 934 行 →
  按 triplets 截齐修复（13,913 对齐校验）→ `--append` 哈希幂等续跑补齐。
  教训固化：脚本快路径在 PIL 解码前做哈希跳过

### B2. DetailCaps：0 条入库（协议 schema 错位，停下待裁决）

实测列名：`GT_Caption_GPT4O / GT_Caption_GPT4V / GT_Caption_Gemini15Pro /
CogVLM / ShareCaptioner / LLaVA_v15 / binary / image / source`——
registry 的 REFERENCE_FIELDS 链**零命中** → 4,870 条全部因无 reference
落 unusable。这不是随机性缺口（seed=43 族可补），而是**实质性映射决策**
（reference 取哪一列），按"原样保真,不得自行改写"纪律停下。
**建议方案（待批）**：reference := 三列 GT_Caption_* 中最长者
（镜像 registry 对 caption 列表"取最长"的既有约定），映射作为
[EXPORT-MOD] 同级的登记性适配写入脚本。批准后 --append 补跑即可
（~10 分钟，登录节点）。

### B3. 双键去重（builder 重建已验证 EXACT_MATCH）

- 旧线 builder 脚本预 git 时代未入库；按工件重建并对旧 35K 输入复算验证：
  **图像键交集 0（=历史）；问题键碰撞在重建口径**（首轮 human 问题、
  去 `<image>` 后 strip、逐 eval 条目计、统计范围 new200/old100 冻结）
  **下 n=50、mean 475.06、11 桶直方图逐条精确吻合**——重建成立
- 本次应用：DOCCI 对 manifest_300 + MM-Vet 218 图像键/内容哈希
  （coco_subset 1,987 + mmvet 218 张本地评估图 sha256 辅助键）：
  **命中 0，零交集断言通过**（DOCCI 自有摄影图库，符合预期）；
  源间互重 0；问题键为固定模板与评估问题无碰撞

### B4. 元数据备份

`archives/longform_fixed_v1_meta.tar.gz`（3.2MB，triplets+manifest，
sha256 `a4e75ffbac48…`）——**请下载一份到本地**。

### B5. 水位

staging 散件已清（−14.8K inode，702.5K→687.7K）；staging 保留官方原始包
（7.59GB tar.gz + descriptions，谱系源）；scratch 4.97/10 TiB。

## 决策点清单（等你/导师）

1. **DetailCaps reference 映射**（B2 方案待批）——批后 ~10 分钟补齐 4,870 条
2. **规模缺口**：现量 14,847（DetailCaps 批后 ~19.7K）vs "30-40K" 口径——
   补占位源（需自备 JSONL）或接受现量
3. **max_new_tokens 512 vs 384**：DOCCI 实测均长 139.9/p90 209——
   384 已覆盖 p99+；512 的额外空间只对模型自由发挥有意义,请结合
   "长文本"叙事定夺
4. rollout 生成（GPU 必审单）与长文本特征抽取（复用 W1 模板）均不在本轮
