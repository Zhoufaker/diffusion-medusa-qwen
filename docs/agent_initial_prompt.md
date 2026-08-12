# Agent Initial Prompt — Linked Medusa Head Project

> **Read this file first, then read `linked_medusa_spec.md`.** This file tells you what to do; the spec tells you how to implement it.

---

请按 `linked_medusa_spec.md` 实现 Linked Medusa Head。

## 第 0 步：环境侦察（只读，不删任何东西）

在动代码之前，先做下面这些**只读**任务，把结果汇报给我，**绝对不要执行任何 `rm` 或删除操作**。

1. **列出 `/g/data/li96/mz9869/` 下所有内容**，每项标注大小（`du -sh`）和最后访问时间（`ls -lu`）。

2. **列出 `/scratch/li96/mz9869/` 下所有内容**，同样标注大小和最后访问时间。

3. **确认 Qwen2.5-VL-7B-Instruct 的实际路径**——在 `/g/data/li96/mz9869/models/` 下 ls 看看有哪些模型目录，找到 Qwen2.5-VL-7B-Instruct 对应的那个，告诉我具体路径。同时检查目录里有没有以下文件之一（用来抽 lm_head 权重）：
   - `model.safetensors` 或 `model-*.safetensors`（HF 格式分片）
   - `pytorch_model.bin` 或 `pytorch_model-*.bin`
   - 还有 `config.json`（确认 `hidden_size: 3584` 和 `vocab_size: 151936`）

4. **完整盘点 `~/medusa-qwen/` 现有内容**——这个目录里我之前跑过不只一个项目（vanilla Medusa、vision-aware Medusa 等），需要先看清楚再决定怎么处理。请做以下事情：
   - 列出**所有层级**的目录结构（不只是顶 2 层），每项标注大小
   - 对每个 `.py` 文件，看一眼文件名 + 文件开头几行 docstring/注释，给我一行简短说明它大概是做什么的
   - 列出所有 `.pbs` / `.sh` 脚本和它们的简短用途
   - 列出 `notebooks/` 下的 ipynb 文件（如果有）和大概用途
   - **这一步只读，不要改任何东西**

5. **检查 `~/medusa-env/` 是否还在、能用**：列出 `pip list` 中跟训练相关的关键包（torch, transformers, accelerate, safetensors）和它们的版本。

6. **检查 quota**：在 home 节点上跑 `lquota` 或 `nci_account -P li96`，把 `/g/data/li96/` 和 `/scratch/li96/` 的当前使用量和上限告诉我。

完成第 0 步后停下来等我确认。

## 第 1 步：归档旧代码 + 建新工作目录（等我确认完第 0 步再做）

我看了截图后发现 `~/medusa-qwen/` 里东西比我以为的多（不只是 vanilla Medusa，还有 vision-aware Medusa、PBS eval 脚本、notebooks 等），所以**不要直接删除**。改用归档方案：

```bash
mv ~/medusa-qwen ~/medusa-qwen-archive-qwen3vl
mkdir -p ~/medusa-qwen
```

这样：
- 旧代码完整保留，零误删风险
- 新项目从干净的 `~/medusa-qwen/` 开始
- 之后想用旧代码里的工具（PBS 模板、eval 脚本、config helper 等），随时可以从 archive 里翻

但**等我看完第 0 步报告再执行**——我可能基于第 0 步内容微调归档名或者要求你保留某些文件直接软链接到新目录。看到我说"OK 执行归档"再做。

如果第 0 步报告显示 `/g/data/li96/mz9869/` 下有可以清理的大文件（比如重复的旧 checkpoint），我会单独给你一份具体的 `rm` 清单。**不要自己判断什么该删**。

## 第 2 步：抽取 base model 的 lm_head 权重

等清理完毕后，写一个一次性脚本 `scripts/extract_base_lm_head.py`：

- 从第 0 步确认的 Qwen2.5-VL-7B-Instruct 路径加载模型（用 safetensors 或者 HF transformers 都行，看哪个不需要把整个 7B 模型加载到 GPU——只需要拿 lm_head 那一层 weight，最好不要加载整个模型）
- 把 lm_head.weight 单独存成一个文件到 `/g/data/li96/mz9869/medusa_assets/base_lm_head.safetensors`（小文件，放 gdata 没问题）
- 验证 shape 是 `(151936, 3584)`

跑一次确认成功后给我看输出。

## 第 3 步：实现代码

当前阶段 cache 还没上传到 `/scratch/li96/mz9869/cached_data/`，所以用 `SyntheticVLMDataset` 跑通代码结构。

实现顺序：

1. 先实现 §5 的三个 module（MLPResBlock, LinkedMedusaHead, LinkedMedusaHeads）+ §6 的 SyntheticVLMDataset/collate/loss
2. 写 `scripts/debug_forward.py` 跑通 §11 Phase A 所有 `(synthetic OK)` 的 check
3. 最后才写完整训练 loop 和 PBS 脚本

严格遵守 §2 的 confirmed decisions 和 §10 的 "Locked" 列表，不要自作主张换设计。

完成 1、2 步后停下来给我看结果，不要直接进第 3 步。

---

## 通用约束（适用于全过程）

- **`linked_medusa_spec.md` 的 §2 confirmed decisions 和 §10 Locked 列表是导师确认过的设计，不是建议**——任何偏离都需要先问我。
- **任何 `rm` / 删除操作都需要我确认后才能执行**，绝不自作主张。
- **任何不在 spec 里的 hyperparameter 或设计决策**，问我，不要自己定。
- **每个步骤完成后停下来报告结果**，不要把多个步骤连起来一口气做完。
- **不要做 §12 列出的 out-of-scope 内容**（tree attention、verification、teacher loss、token embedding passing 等）。

---

## 方法论沉淀

### 探针/门失败 → 停等复核，不得自行修复重跑

**原则.** 任何 probe / gate / acceptance 检查报失败时，正确动作是**停下、如实上报、等待复核**。即使失败看起来明显是探针侧缺陷而非被测对象缺陷，也不得先自行诊断、打补丁、重跑，事后再补报结果。理由：判定"这是探针的错，不是结论的错"本身就是需要被审的科学判断；自证自纠会让"失败被修掉"和"失败被解释掉"在证据链上不可区分。

**实例（2026-08-08，candidate-specific near-tie re-probe）.**
第一次探针 `175812069` 在 512 条首次中段分歧里报出 21 条 `hard`。当时的处理是：自行定位到探针侧缺陷（OOD 组未套用 `max_pixels=501760`，导致重放的 greedy 前缀漂移）、修好（按 tag 设 `max_pixels` + 新增 context-fidelity 门）、重跑为 `175813855`（512/512 `near_tie`、512/512 上下文校验通过），**然后才**上报。修复本身正确、结论也站得住（21 条失败 21/21 落在 OOD 段、in-domain 段零失败，错因闭合），但流程是越界的——应当在报出 21 条 `hard` 的那一刻就停。

**沉淀.** 该案已记入 `c1_eagle_conditioning.md` 的 process change log（第 3 条），与第 1 条"canonical 脚本改动须先报批"同属一类：**涉及证据链的动作，先报批，不先动手。** 缺陷探针的产物必须原样归档（本例：`round6_candidate_reprobe_v1_nomaxpixels/`），不得被成功重跑覆盖。
