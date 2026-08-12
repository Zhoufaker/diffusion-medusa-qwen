# 导师代码包清点报告 — fixed_data_code_pack_local_20260811_203959

- 清点日期：2026-08-12
- 清点范围：只读、静态核对；未运行任何训练/数据生成命令，未加载模型
- 包存放：`/scratch/li96/mz9869/external/mage_pack/`（原始 tar.gz 已设为只读 444）
- 项目内入口：`~/medusa-qwen/external/mage_pack` → 软链接指向上述 scratch 目录

---

## 0. 执行摘要（三条最重要的发现）

1. **包内没有任何 block diffusion / DFlash 相关内容。** 全包 grep `diffusion|dflash|block.diff|denois` 零命中（唯一命中是 `abl_chained_variants.py:183` 的布尔 mask 注释，无关）。这是导师 **Medusa/EAGLE-chained 旧线**的代码包：5-head ResBlock drafter + flat/tree 验证 + 若干消融。
2. **导师的 target model 是 LLaVA-1.5-7B，不是 Qwen2.5-VL。** 训练与消融脚本硬编码 `MODEL_ID = "llava-hf/llava-1.5-7b-hf"`；仅 `task_42_sd_cost_profile.py` 一个脚本可选 `qwen25vl`。他引用的 84K 训练缓存与我们的 34,999 条 Qwen2.5-VL TARGET convention 缓存**谱系不一致**（不同 target、不同 prompt 分布、不同回答形态）。
3. **包内不含数据与权重本体。** README 明言"intentionally excludes checkpoints and caches"；所有 checkpoint / cache 路径均指向导师实验室机器（lab0/lab1，`/home/yxma/...`），在 Gadi 上不可访问。"fixed data" 的实际含义是**数据固定协议 + 导出脚本**，不是一份可直接使用的数据。

---

## 1. 解压与存放（任务步骤 1）

- 原始包：130,339 bytes（≈127 KB），解压后 110 KB / 44 个文件。**无大文件**（最大单文件 92 KB 的 `scheduler_queue.json`），不存在 >10GB 情形，已全量解压。
- 解压后 lquota（li96 scratch）：**4.15 TiB / 10.00 TiB，inode 964,286 / 1,018,000**。本次操作增量可忽略；注意 inode 已用 94.7%，接近限额（与本包无关，为既有状态，见 §6）。
- 附带事故处理：建 `~/medusa-qwen/external/` 时触发 **home 配额超限**（10262M/10240M）。定位为 `~/.cache/pip` 占 128M（违反项目"home 不放缓存"纪律），已删除该 pip 下载缓存（可安全再生），现 home 10135M/10240M，**余量仅约 105M**，见 §6 建议。

## 2. 基座模型完整性核查（任务步骤 2）——结论：**静态完整**

**a. 实际路径**：`/g/data/li96/mz9869/models/` 为**空目录**（历史清理属实，gdata 上无模型）。完整模型位于 scratch 的 HF hub 缓存：

```
/scratch/li96/mz9869/tmp_hf_download/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/
  snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/
```

**b. 分片核对**（index 登记 5 片，磁盘 5 片，逐一比对）：

| index.json 登记分片 | 磁盘存在 | 字节数 |
|---|---|---|
| model-00001-of-00005.safetensors | ✓ | 3,900,233,256 |
| model-00002-of-00005.safetensors | ✓ | 3,864,726,320 |
| model-00003-of-00005.safetensors | ✓ | 3,864,726,424 |
| model-00004-of-00005.safetensors | ✓ | 3,864,733,680 |
| model-00005-of-00005.safetensors | ✓ | 1,089,994,880 |

分片合计 16,584,414,560 B；index `total_size` = 16,584,333,312 B（tensor 数据净大小，不含 safetensors 头，差值 ~81 KB 属正常）。无缺片、无零字节片，量级符合 7B bf16 ≈16.6 GB。

**c. 配套文件**：config.json、generation_config.json、tokenizer_config.json、tokenizer.json、vocab.json、merges.txt、preprocessor_config.json、chat_template.json **全部在位**。抽 feature 必需的 `preprocessor_config.json` 确认存在。

**d. 结论**：**静态完整**。实际可加载性按计划留待 D1 首个作业顺带验证。另注：`/scratch/li96/mz9869/medusa_assets/base_lm_head.safetensors`（1.09 GB，旧线"只抽 lm_head"产物）仍在，与完整模型并存，不冲突。

**e. 清除风险提示**：分片 atime 为 2026-08-09（3 天前，安全）；但 `merges.txt`/`vocab.json` atime 停在 2026-05-12（92 天），距 scratch 100 天自动清除仅剩约 8 天。本次清点已通过读取将其 atime 刷新至今日。**该模型目录属长期资产，建议本地备份或定期 touch**（16.6 GB 重下并不贵，但会阻塞 D1 排期）。

## 3. 目录树与文件清单（任务步骤 3）

```
fixed_data_code_pack_local_20260811_203959/
├── README_FIXED_DATA_CODE.md          # 包总说明：内容、lab0 布局、smoke/full 命令
├── metadata/
│   └── file_manifest.txt              # 44 个文件的原始路径清单（与实际内容核对一致）
├── code/                              # 8 个核心脚本
│   ├── longform_dataset_registry.py   # 数据固定核心：确定性加载 9 类长文本 VLM 数据集，归一化为统一 sample 结构
│   ├── task_45_longform_dataset_export.py  # 7 行入口，调 registry.main() 导出 JSONL 清单
│   ├── task_41_fair_compare.py        # ViSpec vs Medusa+Tree vs Chained+Tree 同协议 walltime 对比（LLaVA-1.5-7B）
│   ├── task_42_sd_cost_profile.py     # SD 成本剖析：prefill/decode/draft/verify 延迟；唯一支持 qwen25vl 的脚本
│   ├── task33_adaptive_tree.py        # 置信度门控自适应树 SD（confidence>阈值走 top-1，否则分支）
│   ├── abl_chained_variants.py        # 训练入口：independent/hydra/full(soft-chain)/nofilter 四变体，5-head drafter
│   ├── abl_hydra_vs_soft_walltime.py  # 三种 head 连接方式的同协议 walltime 对比 runner
│   └── conditional_filtering_nll_suite.py  # 条件过滤 + reached-position NLL/CE 诊断套件
├── ablations/
│   ├── hydra_vs_soft_chain/           # run 脚本、checkpoints.json（ckpt 均在 lab0）、protocol_manifest.json、smoke 结果
│   ├── exactness_check/               # run_exactness_check.py（AR vs SD greedy token 逐位一致性校验器）+ 4 组小结果
│   └── conditional_filtering_nll/     # run 脚本 ×3 + smoke/诊断结果（诊断标记 not_run_missing_local_artifacts）
└── docs/home/yxma/.../vlm-speculative-calibration/   # 导师工作区文档快照
    ├── experiment_plan.md / experiment_notes.md / experiment_results.md  # 实验计划/详录/结果
    ├── iter_009/paper/                # 论文用消融汇总与实验章节草稿
    └── scheduler_queue.json           # lab 集群任务队列快照（含各 cache/训练命令原文）
```

**shell 脚本**（一句话）：`run_flat_walltime.sh` / `run_conditional_filtering_walltime.sh` / `run_nll_diagnostic.sh` / `run_smoke.sh` 均为对应 python runner 的环境变量包装（GPU、N_SAMPLES、数据集选择）；`launch_lab0_wait_run.sh` 是 lab0 上等 GPU 空闲再启动的调度包装。

**json**（一句话）：`checkpoints.json` = 三个消融 ckpt 的 lab0 路径与定义；`protocol_manifest.json` ×2 = 预注册的协议（数据集、verifier、指标、资源）；`results/*.json` = smoke / 小规模运行结果；`scheduler_queue.json` = 任务队列记录。

**数据文件**：包内**无训练数据文件**。各 `results/*.json` 为结果记录，抽样示例（`exactness_check/run_lab1_n1/exactness_results.json`）：

```json
{"config": {"target_model": "llava-hf/llava-1.5-7b-hf",
            "draft_info": {"draft_type": "chained_eagle_steps", "hidden_size": 4096,
                            "n_heads": 5, "n_layers": 2},
            "datasets": ["detail", "docci"], "max_new_tokens": 16, "dtype": "float16"},
 "per_sample": [{"dataset": "detail", "mode": "flat", "exact_match": true,
                  "first_mismatch_pos": null, "ar_len": 16, "spec_len": 16,
                  "ar_tokens": [450, 1967, ...]}]}
```

## 4. 重点问题逐条回答（任务步骤 4）

### 4a. 数据固定脚本的输入是什么？

- **prompt 源**：数据集自带 question 字段，缺失时用固定模板。证据：`longform_dataset_registry.py:26-30` 定义 `DEFAULT_PROMPT`（"Describe this image in detail."）与 `LONGFORM_PROMPT`；`:167` `question = _field(record, QUESTION_FIELDS) or default_question`。
- **图像源**：两路。(1) HF datasets 在线加载：`detail` → `lmms-lab/LLaVA-Bench-In-the-Wild`（`:367-378`，filter `category=="detail"`）、`detailcaps` → `DetailCaps-4870`（`:380-392`）、`docci` → `google/docci`（`:394-405`）；(2) 本地 JSONL/JSON + image_path（`load_jsonl_samples`, `:226-252`）。另有 4 个数据集（localized_narratives / vqaonline / stanford_paragraph / vizwiz_lf）只留了报错占位，要求用户自备本地 JSONL（`:407-431`）。
- **是否依赖 target model rollout**：**不依赖**。全文件无任何模型 forward；固定的是 (image, question, reference) 三元组与采样顺序（`seed=42` shuffle，`:236,331`）。reference 取数据集人工/原始长答案（`:168`）。

### 4b. 固定产出的数据是什么形态？

- **JSONL 清单 + 逐张 JPEG 图片目录**，不是 token 序列，也不是 hidden feature。证据：`write_samples_jsonl`（`longform_dataset_registry.py:436-453`）每行写 `{id, source, image_path, question, reference}`，图像另存为 `{idx:06d}.jpg`（quality=95）。
- 注意：**逐张 jpg 落盘正是我们 inode 事故的模式**（CLAUDE.md 存储纪律第 2 条）。若在 Gadi 上跑该导出，n 大时需改为打包格式或落到低 inode 压力位置。
- 训练消费的则是另一形态：每样本 `.pt` 文件 `{"hidden": [N,D], "tokens": [N]}`（`abl_chained_variants.py:138-140` `CachedDataset.__getitem__`），即 **target hidden-state feature 缓存**。**从 JSONL 清单到 .pt 缓存的生成脚本不在包内**（scheduler_queue.json 中可见 `code/cache_ov15.py` 等命令原文，但脚本本体未打包）。

### 4c. 包内是否含已生成数据？谱系是否与我们一致？

- **包内不含已生成数据**（README_FIXED_DATA_CODE.md:5-8 明言排除；全包最大文件 92 KB）。
- 代码引用的缓存为 **84K short-answer VQA cache**，路径 `lab0:~/vlm_sd_exp/cache/llava_scaled/`（`abl_chained_variants.py:46`；`experiment_notes.md:15`；`experiment_plan.md:54`），源数据集为 TextVQA/ChartQA/ScienceQA/AI2D/OKVQA/LLaVA-Bench 等短答 VQA（`experiment_plan.md:69`）。target model 为 **LLaVA-1.5-7B**（`checkpoints.json` `"target_model": "llava-hf/llava-1.5-7b-hf"`）。
- 生成方式（greedy rollout vs teacher-forcing）**包内无一手代码证据**（生成脚本缺失）；文档侧 `experiment_plan.md:565` 记录评估协议为 greedy decoding，但训练缓存的构造方式待向导师确认。
- **与我们 34,999 条缓存的谱系判定：不一致**。四个维度全部不同：target（LLaVA-1.5-7B vs Qwen2.5-VL-7B）、规模（84K vs 34,999）、prompt 分布（短答 VQA vs 我们的 self-distillation 长回答 rollouts）、hidden 维度（4096 vs Qwen2.5-VL 的 3584）。他的缓存对我们**不可直接复用，也无法在 Gadi 取得**。

### 4d. 代码对训练框架的假设？与 Gadi 环境是否冲突？

- **依赖库**：torch、numpy、transformers（`DynamicCache`，`task_41_fair_compare.py:29`）、datasets+pyarrow（`longform_dataset_registry.py:255-260`）、Pillow、requests、wandb（`abl_chained_variants.py:25`，训练脚本硬 import，Gadi 计算节点无外网，需 offline 模式或去除）。无 requirements 文件、无版本 pin；唯一版本线索：tree attention 需 **HF transformers ≥4.50 的 4D attention mask**（`task_41_fair_compare.py:18-19` 及 `:592` `attn_implementation="eager"`）。
- **不用 FlexAttention**。attention 走 HF `attn_implementation` 参数：默认 eager（4D mask 需要），`task_42_sd_cost_profile.py:769` 可选 `eager/sdpa/flash_attention_2`。无 `torch.compile`、无 DeepSpeed/FSDP，单卡朴素训练循环。
- **GPU 假设**：单卡 `cuda:0` 硬编码（`abl_chained_variants.py:27`）；target fp16（`:377`）+ drafter fp32（`abl_hydra_vs_soft_walltime.py:450`）；walltime 协议要求 ≥22GB 独占 GPU（`protocol_manifest.json` `"gpu_vram_gb": ">=22"`，导师侧用 RTX 4090/A6000）。lab0 的 CUDA 编号 quirk（README:106-110）为其本地问题，与 Gadi 无关。
- **与 cuda/12.3.2 冲突判定**：无编译扩展、无 flash-attn 强依赖，纯 PyTorch/HF 代码，**原则上不冲突**。实际风险点是我们 `requirements.lock` 的 transformers 版本是否 ≥4.50（tree 验证需要），以及 wandb 在 Gadi 上的联网问题——都属接入时的小改造，非硬冲突。硬编码的 lab 路径（`~/vlm_sd_exp/...`、`/mnt/sda/...`、`task_41_fair_compare.py:35`）在 Gadi 上全部无效，凡复用必须改路径（且按纪律先复制出包再改）。

### 4e. 是否包含 diffusion drafter 的模型定义或训练循环？

- **不包含。** 全包无 diffusion/DFlash/block-denoising 内容（grep 零命中）。
- 包含的是 **Medusa/EAGLE-chained drafter**：`MedusaVariants`（`abl_chained_variants.py:68-111`）——**5 heads × 2 层 ResBlock**（Linear(D,D)+SiLU 残差，`:59-65`），hidden_size 取自 target config（LLaVA 为 4096，`:386`）。feature 注入方式即四变体的区别：
  - `independent`：各 head 只读 `hidden + token_embed`（`:90-93`）
  - `full`（soft chain）：head k 加上 head k-1 的 **detached hidden feature**（`:107`）
  - `hydra`：head k 加上 `embed(argmax(lm_head(prev_feat)))` 离散 token 链接，不可微（`:99-105`）
  - `nofilter`：同 full 但关闭 conditional filtering（`:49-54`）
- 训练循环：逐 head 顺序训练、前序 head 冻结（`:245-256,346`），loss 为 lm_head 上的 CE（`chunked_lm_head_loss`，`:143-149`），conditional filtering 用前序 head 全对的位置 mask（`compute_acceptance_masks`，`:182`）。30 epochs/head、BS32、LR 3e-5×√BS、cosine+5% warmup（`:33-39`）。**无 block size 概念**——这不是 block diffusion 架构。

## 5. 附带发现与风险（任务步骤 1/2 过程中）

1. **home 配额事故（已处置）**：清点开始时 home 已超限 10262M/10240M，`mkdir` 失败。删除 `~/.cache/pip`（128M，pip 下载缓存，可安全再生）后恢复。当前余量仅 ~105M；home 大头为 `medusa-env`（5.8G）与 `.cursor-server`（3.9G）。**建议**：venv 迁往 gdata 或 scratch（venv 属可重建资产，scratch 100 天风险可接受）择机执行，避免下次在作业中途爆掉。
2. **scratch inode 已用 94.7%**（964k/1018k）。导师的数据导出（逐张 jpg）与训练缓存（每样本一个 .pt，84K 文件量级）都是 inode 密集型模式，**照搬到 Gadi 前必须改打包格式**（tar/webdataset/safetensors 分片）。
3. **模型 tokenizer 小文件 atime 距 100 天清除仅 ~8 天**（已刷新，见 §2e）；建议把整个模型 snapshot 列入"建议本地备份"清单。

## 6. 接入建议（【建议】，最终由你判定）

**判定：并行（作为协议参考），不替代、不补充。**

理由：

1. **谱系不兼容，无替代可能**：他的数据固定针对 LLaVA-1.5-7B 的短答 VQA 缓存（84K，hidden 4096），我们的 34,999 条是 Qwen2.5-VL-7B greedy rollouts（TARGET convention，hidden 3584）。target 不同则 hidden feature 缓存完全不可互换；prompt 分布（短答 vs 长文本）也与我们锁定的三层评估体系不匹配。且数据本体根本不在包里、留在导师实验室机器上。
2. **也不宜"补充"**：把 LLaVA 谱系样本混入 Qwen 缓存违反单变量消融纪律，且会污染已锁定的评估谱系（34,999 条与评估图像双键去重的前提被破坏）。
3. **真正有价值、值得并行采用的是三样"协议资产"而非数据**：
   - `longform_dataset_registry.py` 的**确定性数据固定协议**（固定 seed、归一化 schema、JSONL 清单化）——可借鉴其形式为我们的 block diffusion 线固定 held-out/OOD 清单（我们已有 seed=43 嵌套设计，主要是核对他的协议与我们无冲突后，向导师对齐"fixed data"口径）；
   - `run_exactness_check.py` 的 **AR vs SD greedy 逐 token 一致性校验**——与我们的字节级回归门同族，可作为新线 verify 正确性的独立 sanity check（需移植到 Qwen2.5-VL）；
   - `conditional_filtering_nll_suite.py` 的 **reached-position NLL/CE 诊断**——与我们"接受率校准"的核心研究问题直接相关，其指标定义（per-head conditional acceptance、reach rate，见 `protocol_manifest.json` metrics 节）值得纳入我们的预注册指标讨论。
4. 他的 Medusa/EAGLE-chained 代码线与我们已降级的旧线同族，按纪律**只读、只作 baseline 语境参考**，不接入训练管线。

若采纳"并行"判定，建议下一步（均为轻量、非训练操作）：与导师确认 84K 缓存的生成方式（rollout vs teacher-forcing）以完整回答 4c；核对我们 `requirements.lock` 的 transformers 版本是否 ≥4.50。

---
*清点人：Claude（只读清点，未运行训练/生成命令；home pip 缓存删除与 atime 刷新为仅有的两处状态变更，均已在 §5/§2e 记录）*
