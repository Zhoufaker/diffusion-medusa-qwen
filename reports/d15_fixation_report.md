# D1.5 长文本数据固定报告（阶段一：盘点与固定）

> **v2 节见文末**（2026-08-20 起 longform_fixed_v2 为现行版本，37,079 条；
> 本文上半部为 v1 历史记录，v1 目录保留只读）。

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

1. ~~DetailCaps reference 映射~~ **已批执行完毕**（2026-08-19，见下方 B2 后记）
2. **规模缺口**：终量 19,715 vs "30-40K" 口径——补占位源（需自备 JSONL）
   或接受现量，等裁决
3. **max_new_tokens 512 vs 384**：DOCCI mean 139.9/p90 209，DetailCaps
   mean 228.8/p90 265——**等 rollout 长度 pilot 实测**（模型自由发挥长度
   才是裁定依据，备审材料已交）
4. rollout 生成（GPU 必审单，pilot 材料已备审）与长文本特征抽取不在本轮

## B2 后记（2026-08-19，决策点①已批执行）

- 映射按批定方案落地（iter_detailcaps shim，登记注释在案）：
  **4,868/4,870 入库**（2 条内容哈希重复剔除；**dup_eval=0——含 700 条
  coco2017 来源图，图像键（COCO-id 兼容基名）与内容哈希对两份评估清单
  实跑均零命中，如实报数**）
- DetailCaps ref 长：mean 228.8，med 229，p10 188，p90 265，max 517
  （≥300 占 2.3%——同样低于 300 线，如实登记）
- **最终规模：19,715 条 / 20 片**；零交集终检（rescan）通过
- 元数据 tar 已重打：sha256 **3c42dba8107e…**——**请重新下载本地备份**
  （旧 a4e75ffb 版作废）
- 收尾插曲：补跑收尾 sanity assert 未兼容 prior-only 统计条目而崩
  （数据零影响——打包先于断言完成）；断言已修，manifest 由 finalizer 重写
  并复跑零交集终检，全程 fixation.log 在案

---

# v2 节（longform_fixed_v2，2026-08-20 定稿）

## 裁决与范围（用户 2026-08-19 下达）
只用 train split 原则；DOCCI 切分修正（全 split 14,847 → **train 9,647**）；
新源 SP + LN-OI；VizWiz-LF 探明不做（600 对 <1K 门槛 + 多为 VLM 生成答案 +
BLV 隐私域）；目标 35-37K；v1 保留只读。执行走用户特批的 copyq 单作业链。

## 终态（job 176678578，哨兵在案）

| 源 | 入库 | dup_eval | ref token 均长/中位/p90 | ≥300 |
|---|---|---|---|---|
| DOCCI(train) | 9,647 | 0 | 139.5 / 130 / 208 | 1.0% |
| DetailCaps | 4,868 | 0 | 228.8 / 229 / 265 | 2.3% |
| SP(train) | **14,564** | **11**(VG↔COCO 血缘实测) | 70.3 / 73 / 98 | 0% |
| LN-OI(采样) | 8,000 | 0 | 39.8 / 36 / 66 | 0% |
| **合计** | **37,079 / 38 片** | 11 | **加权均长 ~103** | — |

- 零交集终检通过;各源抽检 5/5 ×4 全过(解码/模板;docci/sp/ln 并对源
  reference 逐字节核对,dc 为构造性保真)
- LN:8,000 采样 CVDF 下载仅 1 败且 Flickr 兜底成功,最终 0 损耗
- SP 事故与纠正:首跑漏配 8,077 张(images.zip 成员无目录前缀,
  匹配键误带前缀;6,502=带前缀侧全中,数字闭合)→ 基名匹配修正后
  纠正重跑(21 分钟,5.72 SU)。首跑 77.19 SU 计入损耗
  (copyq 按内存折算 ~16 SU/h,前估 15 严重偏低,教训登记)
- **⚠ 长文本属性进一步稀释(如实报)**:池子加权均长 v1 ~162 → v2 **~103**
  (SP 70.3、LN 39.8 拉低)。max_new 裁定与"长文本"叙事请结合此实测
- 元数据 tar:archives/longform_fixed_v2_meta.tar.gz,
  sha256 **f1aee1a6e47f…**——**请下载本地备份(3c42dba8/27fb361e 版均作废)**
- SP 标注来源:官方 zip 的 wayback 20230307 快照(19,561 段,
  切分 14,575/2,487/2,489 与公开统计吻合);VG 图像官方 zip 15.2GB
- 许可:DOCCI CC BY 4.0;SP/VG CC BY 4.0;LN 标注 CC BY 4.0 +
  OI 图像 CC BY 2.0(Flickr 源);DetailCaps 见数据卡

## 送审件(本块唯一送审出口)
rollout 长度 pilot v2 版:50 条按源占比分层(docci 13/dc 7/sp 19/ln 11,
seed=43),gen_cache_rollout.py 零改动 + prep 适配层,--max-new 512,
dgxa100 30min/<10 SU。PBS: pbs/d15_rollout_pilot.pbs。等审。
