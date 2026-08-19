# REVIEW_NOTES — d15_gen_rollouts 批量生成必审包

审查对象:37,079 条 longform rollout 批量生成(队列 ①)。
包内文件:scripts/gen_longform_rollouts.py(代码)、
tests/test_gen_longform_rollouts.py(CPU 单测,19/19 过)、
pbs/d15_gen_rollouts.pbs(array 草案)、reports/d15_gen_rollouts_budget.md
(预算,已按实测回填)、reports/d15_rollout_pilot_report.md(pilot v1+v1.1,
吞吐锚与 spec 动因)。行号以包内文件为准(commit 见 git log)。
必审触发:新脚本首次上卡 + 写入新位置 + 预估 SU 2,581 > 500。

---

## A. gen_longform_rollouts.py(六处自查点)

### A1. 批内 EOS 截断与"EOS 是否计入序列"口径
- 实现:`trim_new_tokens` **L171-178**——首个 EOS【计入】保存序列,
  之后(generate 的 pad 填充段)丢弃;无 EOS 达 max_new → 全长保留 +
  `eos_hit=false` 登记。调用点 **L236-237**(`out[b, p_len:]` 切片,
  p_len=左 pad 后统一 prompt 长,批内各行同值)。
- 旧线比对(gen_cache_rollout.py):prefill 首 token 入列 **L53**
  (`toks = [tok]`);循环条件 **L56**(`while steps < max_new and
  tok != eos_id`)——先 append(**L63**)后判,故 EOS 作为末 token 计入,
  序列长 ≤ max_new。
- **结论:两处口径一致**(EOS 计入、上限含义相同)。附注:TARGET
  convention 的 hidden↔token 对位是队列 ② 抽取层的职责,本脚本只产
  tokens,不触及。

### A2. left-pad 与 M-RoPE 路径(走模型原生 rope_deltas)
- `processor.tokenizer.padding_side = "left"` **L303**;运行时守卫
  `assert_left_padded` **L160-168**(mask 逐行单调 0…1 + mask==0 处
  必为 pad + 无全 mask 行),每个 batch 都过(调用点 **L229**)。
- 生成调用 **L232-235** 只传 processor 标准输出 + generate 标准参数,
  **不传显式 position_ids** → 走模型原生 M-RoPE:transformers 5.3.0
  modeling_qwen2_5_vl.py **L1077** `get_rope_index`,位置只在
  `attention_mask.bool()` 选中的位置上计算(**L1150-1151, L1181**,
  左 pad 安全),rope_deltas 由模型缓存并在 decode 续用(**L1291-1311**)。
- decode/common.py **L203-211**(continuation_base docstring)警示的
  "显式 position_ids 必须加 rope_delta 偏移否则图像 prompt 位置错
  数百"陷阱:本脚本因不传 position_ids 而整体绕开。
- 单测:test_left_pad_ok / test_right_pad_rejected /
  test_interior_pad_rejected / test_pad_inside_attention_rejected /
  test_fully_masked_row_rejected。

### A3. greedy 三对齐(与 bs=1 谱系)
- **L232-235**:`do_sample=False, num_beams=1, repetition_penalty=1.0,
  eos_token_id=eos_id(单值 151645), pad_token_id`。
  动因:模型 generation_config.json 自带 `repetition_penalty: 1.05`、
  `eos_token_id: [151645, 151643]`、`do_sample: true`——三项不显式
  覆盖即与 bs=1 裸 argmax 分叉(rep_penalty 是 logits processor,
  与采样开关无关,必须显式压回)。
- 幻影词表屏蔽:`PhantomVocabMask` **L292-297**(ids ≥ EFFECTIVE_VOCAB
  → -inf)。等价性:decode/common.py `mask_phantom_` **L41-46** +
  `argmax_masked` **L49-51** 即"置 -inf 后 argmax";greedy 下二者数学
  等价(-inf 项不可能是 argmax,其余 logits 不变)。常量一致性双保险:
  运行时断言 **L300**(`assert EFFECTIVE_VOCAB == C.EFFECTIVE_VOCAB`)+
  单测 test_effective_vocab_matches_decode_common。
- 登记:批内 fp16 数值扰动属已知类别(artifact 锁定制,docstring
  **L6-11**);验证方案含 50 条 pilot 行批量 vs bs=1 对照标定(非门)。

### A4. max_pixels=501760 生效点与 manifest 字段
- 默认值 `MAX_PIXELS_SPEC = 501760` **L64**,argparse 默认 **L261**;
  生效点 **L302** `C.apply_max_pixels(processor, args.max_pixels)`
  (decode/common.py **L495-505**:min_pixels 3136 不动,仅收 max 侧
  ——与 MM-Vet OOD 同一实现,PREPROCESS_SPEC 正典路径)。
- manifest 登记:config dict **L277-282** 含 `max_pixels` 及 greedy/
  rep_penalty/eos/phantom_mask 全套口径回声,由 `build_manifest`
  **L181-187** 写入 manifest.json。

### A5. 续跑幂等的哈希键构造
- 哈希键 = `token_sha256` **L73-78**:sha256(逐 token int64 little-
  endian 8 字节),只覆盖 token 序列本身;口径登记于 manifest
  `token_hash` 字段 **L187**。身份键 = 行内 `idx` 字段(不参与哈希,
  由 schema 校验 + 重复剔除保护)。
- 恢复路径:`scan_shard_file` **L102-123**(截断尾行丢弃、坏 JSON
  丢弃、`validate_record` 逐行复验 sha、重复 idx 保首行,任一情形
  置 dirty)→ `repair_and_load` **L125-137** 仅 dirty 时原子重写
  (tmp + os.replace)。**幂等保证:完好文件字节不变**(单测
  test_resume_clean_file_untouched / _truncated_tail_dropped_then_
  idempotent / _bad_sha_and_duplicate_dropped / _two_phase_completion)。

### A6. JSONL schema 字段清单
- 行 schema = `RECORD_KEYS` **L65**:`idx`(int,全池行号)/
  `sample_id`(str,谱系键)/ `n_tokens`(int,=len(tokens))/
  `eos_hit`(bool)/ `tokens`(list[int],含 EOS)/ `sha256`(hex)。
  构造 `make_record` **L81-84**,校验 `validate_record` **L87-99**
  (字段恰为六键、类型、计数、sha 四重)。
- 总 manifest schema = `build_manifest` **L181-187**:kind / total /
  n_cached / complete / config(口径回声)/ shard_files(逐文件
  n+sha256)/ token_hash。单测 test_manifest_schema_and_complete_flag、
  test_record_schema_violations。

## B. pbs/d15_gen_rollouts.pbs(三处自查点)

### B1. array 0-7 分片边界归属
- `#PBS -J 0-7` **L27** 与 `NUM_SHARDS=8` **L37** 一致,shard id 取
  `PBS_ARRAY_INDEX` **L38**。归属规则 = 脚本 `shard_range` **L140-147**
  连续区间(37,079 = 7 片×4,635 + 1 片×4,634,块差 ≤1,tar 局部性);
  单测 test_shard_range_partition 验 8 片无缝无重叠覆盖 [0,37079)。
- 哨兵回报实际边界(`rows: [start, stop)`,PBS **L64**),审查可与
  预期对账。

### B2. 每 shard 哨兵 + 邮件
- 邮件:`-m ae` **L24** + `-M` **L25**,array 每个子作业独立发信。
- 哨兵:**L55-69**,逐子作业写
  `JOB_DONE_${PBS_JOBID%%.*}.json`——array 子作业 PBS_JOBID 形如
  `176xxxxxxx[3].gadi-pbs`,截断后保留 `[3]`,8 个哨兵文件名互异不
  覆盖。内容为 shard 文件**实态**(用 scan_shard_file 哈希优先重扫,
  L61,而非进程自报计数),含 n_valid/n_expected/dirty_seen/complete。

### B3. 中止预案的门位置(确认实现)
- **确认:流程门,非脚本内自动门**。本 PBS 只含全量 array;冒烟是
  独立单作业另行 qsub(同一脚本 `--limit 64` + bs=8/bs=1 配对,独立
  out-dir,预算文档"验证方案"第 1 步),E 实测落哨兵后**人工确认
  E≥1.5,才手工 qsub 本 array**——两次 qsub 之间天然人工门,array
  文件内无任何自动放行/自动提交逻辑。E<1.5 → 不提交 array,回预算
  单改预案(bs 调参或改多 GPU 并行拆分)。

## C. budget(两处自查点)

### C1. E 的测量口径(选定:同条配对,弃 R₁ 外推)
- 位置:budget **L45-50**(冒烟条目)与验证方案第 1 步。
- **选同条配对**:冒烟对同一 64 行(池首 64 行)同作业同卡先 bs=8
  后 bs=1,E = R_bs8 / R_bs1。
- 理由:冒烟行 = 池首 64 行,**全 DOCCI**(池按源分块排序),与 R₁
  的测量行集(pilot 分层 50 条,四源配额)构成不同;若用 R₁ 外推口径
  (E = R_smoke/R₁),行集与长度构成差异会混进 E。配对口径把 E 纯化
  为批量化增益,代价为 bs=1 臂 ~11 min(已计入冒烟预算 ~25-30 SU)。
  全量预算外推仍用 R₁×E:R₁ 代表全池分层构成,E 是纯批量因子。

### C2. 三个 SU 数字的算式(budget L38-44)
- **2,581**(E=3 保守基准)= 37,079 行 ÷ (0.0958 row/s × 3)
  = 129,014 s = 35.84 GPU·h × 72 SU/GPU·h(dgxa100)
- **1,290**(E=6 乐观)= 37,079 ÷ (0.0958 × 6) = 17.92 GPU·h × 72
- **4,032**(墙钟绝对上限)= 8 shard × 7.0 h(walltime 请求,含 1.5
  安全系数)× 72——PBS 按实际走时计费,预期落 1,300–2,600 区间
- R₁ = 0.0958 row/s 出处:pilot v1.1 追記(job 176694274,50 条/
  8.7 min 生成段,dgxa100 bs=1,新 spec)

---

复核方式:单测 `python -m pytest tests/test_gen_longform_rollouts.py -q`
(登录节点 CPU,~6s);全部行号可用包内文件直接对照。
