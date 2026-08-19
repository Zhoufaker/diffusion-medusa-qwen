# CLAUDE.md — medusa-qwen 项目规约

## 环境（NCI Gadi, user mz9869, project li96）
- qsub 必须带存储指令：`-l storage=gdata/li96+scratch/li96`
- Python 环境：`module load python3/3.11.0 cuda/12.3.2` 后 `source ~/medusa-env/bin/activate`
- 加参数/重训练实验默认 dgxa100 队列（V100 不足以支撑 5-head fp32 训练，历史已验证）；V100 仅用于 OOD 评估门（锚定硬件）

## 当前主线（2026-08 起）
- 主线：Qwen2.5-VL-7B-Instruct 上的 block diffusion drafter（DFlash 式架构，ICML 2026），核心研究问题是接受率的校准
- 旧线（Linked Medusa / 动态树）已降级为对比 baseline，只读不改
- 导师代码包 fixed_data_code_pack_local_20260811_203959.tar.gz 为协议参考资产
  （数据固定协议 / exactness 校验器 / reached-position NLL 诊断），目标 target 为
  LLaVA-1.5-7B，数据与代码不接入训练管线；diffusion 架构参考对象为 DFlash 开源仓库

## 锁定的评估体系（不得改动）
- 三层评估：训练分布（self-distillation rollouts）→ in-domain 300-prompt held-out → MM-Vet OOD 218 prompts
- in-domain 300 为嵌套设计（seed=43），其中旧 100 条为字节级回归门
- MM-Vet 评估固定 max_pixels=501760，在 V100 上跑
- PREPROCESS_SPEC 双规并存（2026-08-20 裁决）：longform 线（rollout 生成/
  特征抽取/longform 评估三处）统一 max_pixels=501760，与 MM-Vet OOD 锚定
  约定对齐；短线旧资产（34,999 缓存及其评估）保持原 spec（无 cap）不动。
  两 spec 不得混用；登记与动因（2.34TB 外推、vision token 占 76.7%）见
  reports/d15_rollout_pilot_report.md 与 d15_fixation_report.md 登记节
- 训练数据：34,999 条 self-distillation 缓存（Qwen2.5-VL-7B greedy rollouts，TARGET convention），与评估图像双键去重
- 计时锚点：pooled greedy 29.918 tok/s（旧线数值；新线 verify 形态不同，计时锚点需重新预注册后才可引用）

## 数据资产锁定
- /scratch/li96/mz9869/onpolicy_data/ —— 目录名有误导性，实际内容为
  自蒸馏谱系源头（rollout_prompts.json + 30,599 张 COCO 图像 +
  LLaVA-Instruct 源 json），是 W1 特征抽取的直接输入。锁定至全量
  抽取+审计通过，期间禁止移动/删除/重命名。冷备 tar 已在 archives/。
- ctx_cache_35k 审计基准（2026-08-14 经 2×2 归因实验重定义，
  实验记录 reports/drift_attribution_report.md）：
  - 门 1【抽取确定性】：同卡同代码重跑 shard_00000 与 pilot 逐字节一致
    （sha256）。已过：05b8b6b7… 双跑一致
  - 门 2【漂移登记】：对 V100 增量基准的位置级一致率如实记录进 manifest
    （pilot 实测 95.93%），不设通过线
  - 已废止：原 99.5%/99.9% 对旧缓存的 exactness 门。理由：Arm1
    （V100+增量、原脚本原样）对旧缓存 100.000% 复现，证明管线无罪；
    4.07% 漂移 = 前向形态（增量 vs 整序列 ~3.9%）+ 硬件（V100→A100
    ~0.16%）的 fp16 数值差，非 bug，不可能通过原门，也无需通过

## 对比 baseline 数值（旧线冠军，只引用不重测；v2 口径）
- 速度基线（v2 e2e paired，ARCHIVE_POLICY_V2_ANCHOR 2026-08-07）：
  static_c1_d3 1.705×；速度冠军 dyn_k8_n24 1.732×
- σ 冠军 dyn_k8_n32：σ=2.841，paired Δ+0.101±0.007（对 static_c1_6432 锚 σ=2.7388）
- MM-Vet OOD realization rate：R≈0.433±0.039
- 附注：static 1.689× 为已废弃的 100-scale segmented 口径，仅出现于
  历史文档与 git tag message，不得用于新线对比
- 基线归档：/scratch/li96/mz9869/archives/linked_medusa_baseline_20260812_r2.tar.gz
  （sha256 19abfff806e1fbc6a55e4149a074f4bf35ee68254af08f62cc866dff45da92ee，
  含 .git 与 tag linked-medusa-final；r1 无 .git 版本保留）
- 待办：冷藏项约 470–500G（非冠军 ckpt / smoke / v1 / phaseA）论文投稿后再清
  （本次判定：不删）
- 长期资产 touch 名单（scratch 100 天 atime，定期访问防清除）：
  - /scratch/li96/mz9869/archives/onpolicy_data_legacy.tar（5.3G，B2 数据冷备；
    散件已删，2026-08-15 收编完成，README_MOVED 在原路径）
  - /scratch/li96/mz9869/archives/llava_general_35k_legacy.tar（33.4G，旧线
    自蒸馏缓存冷备，2026-08-15 三重验证后收编，散件已删）
  - /scratch/li96/mz9869/archives/qwen25vl_long_v1cache.tar（~77G，v1 三头训练
    缓存冷备，2026-08-15 验明正身后收编；原"新线资产"登记有误，实为 v1 缓存
    ——见 w1_full_extract_report.md 附录）
  - /scratch/li96/mz9869/dflash_data/ctx_cache_35k/（616G，训练期高频访问；
    训练间歇期注意 atime）

## 工作纪律
- 预注册优先：所有阈值、统计检验在数据到来前锁定
- 单变量消融；训练算力花费前必须过代码审查
- 任何对导师代码包内文件的操作一律先复制到本项目目录，原包只读
- 目录名不是谱系证据——onpolicy_data 与 qwen25vl_long 两起误标事故的教训；
  任何资产的处置决策前必须实物盘点
- 模块命名与 sys.path 纪律（2026-08-16 冒烟事故教训，scripts/train.py
  遮蔽 train/ 包）：scripts/ 下新文件 basename 不得与仓库顶层包/目录同名
  （现存碰撞仅 scripts/train.py，旧线入口不改名）；任何代码将 scripts/
  加入 sys.path 一律 **append**、严禁 insert(0)；跨包惰性 import 前
  先行 eager import 目标包

## PBS 作业分级审查规则
- 免审（agent 按自查清单核对后可直接 qsub）：
  - 使用已过审模板、仅改动索引范围/输入清单/作业名的重复作业
  - 预估 SU ≤ 500，且只写入已批准的输出目录
  - 不含任何删除/移动既有数据的操作
- 自查清单（免审作业 qsub 前逐项核对并在汇报中列出）：
  -P li96 / 正确队列 / storage 指令 / offline 三件套 /
  HF_HOME+PIP_CACHE_DIR / walltime 与吞吐外推匹配 / SU 预估 /
  输出目录已批准 / 大写入前 lquota
- 必审（发脚本与预算、过审后才可 qsub）：
  - 任何新脚本或新入口参数的首次上卡
  - 训练作业（一律必审，不论金额）
  - 预估 SU > 500，或写入新位置，或触碰锁定资产，或含删除操作
  - 队列/硬件变更（如 V100↔A100）
- 排队/运行中作业所引用的脚本与输入文件一律冻结，不得编辑——
  PBS 在运行时才读取工作区代码，提交后修改会导致"过审代码"与
  "实际执行代码"不一致。需要改动时二选一：等作业结束，或 qdel 重排。

## 存储纪律（历史事故驱动，严格执行）
- 默认一切新数据、输出、缓存写 scratch（/scratch/li96/mz9869/），不写 gdata
  - gdata（/g/data/li96/）配额仅 ~10G 且多次爆过，只放基座模型和少量核心评估资产，未经我确认不得新增
- HuggingFace / torch 缓存必须重定向到 scratch：
  export HF_HOME=/scratch/li96/mz9869/tmp_hf_download/
  严禁写入 ~/.cache/（home quota 很小，历史上塞爆过）
- pip 缓存同样重定向到 scratch：
  export PIP_CACHE_DIR=/scratch/li96/mz9869/tmp_pip_cache/
- ~/.cursor-server 属可安全删除项（误删活跃版本仅触发下次连接时重新下载约 500MB，
  无数据损失）；home 吃紧时优先清理其旧版本目录（bin/linux-x64/ 下非最新 mtime 的目录）
- home 治理（2026-08-20 落实，事故：home 曾爆至 10.57G/10G 硬限）：
  - Claude Code CLI 旧版本二进制在 ~/.claude/remote/ccd-cli/（每版 ~300M），
    home 吃紧时删非当前版本（先 ps 确认在用版本再删）；DISABLE_AUTOUPDATER=1
    已入 ~/.bashrc，版本更新改手动择机
  - ~/.bashrc 已重定向：XDG_CACHE_HOME 与 NPM_CONFIG_CACHE → scratch
    （兜住 torch hub/matplotlib/npm 等杂项缓存）
  - ~/.claude/settings.json 已设 cleanupPeriodDays=7（会话记录 7 天自动清；
    projects/ 现仅 ~7M，警戒线 500M）
- scratch 两个已知风险，写入前评估：
  1. 100 天未访问自动清除——长期资产（如 base_lm_head、固定数据）需在报告中标注
     "建议本地备份"；训练期间高频访问的文件不受影响
  2. 项目级 inode 限额——大量小文件（如逐张图片、逐条 feature 文件）曾导致
     inode 超限、作业被 Held。批量小文件优先打包为单文件格式
     （tar / webdataset / 单个 safetensors 分片）
- 大体积写入（>50G）前先跑 lquota 报告余量，写入后再报告一次
- 2026-08-13 起项目 inode 处于警戒状态（软限 1,018K，一度余 13K）；
  每个作业块收尾必须报告 inode 水位，突增时暂缓大批量文件写入并上报
- 登录节点默认无 Python 环境，任何 python 命令前先 module load + activate（见环境节）

## 作业等待策略（token 节约）
默认：提交 qsub 后不轮询。报告 job id 与预计时长后即结束当轮，
等用户手动告知"作业完成"再恢复。恢复时先读结果哨兵文件（见下），
不要从头 tail 大日志。

零 token 辅助（写作业时落实，不靠轮询实现）：
1. PBS 邮件通知：所有作业加
   #PBS -m ae
   #PBS -M 920928763@qq.com
   作业结束/中止时 Gadi 自动发邮件，用户据此回来提醒。
2. 结果哨兵：每个作业脚本收尾必须写
   <输出目录>/JOB_DONE_<jobid>.json
   内容为单行摘要（exit code、关键指标、产物路径、异常计数）。
   恢复会话时 agent 只读哨兵 + 按需抽查日志，不全文加载 .OU。

例外——允许有限轮询，仅当同时满足：
- 预计总等待（排队+运行）< 20 分钟，且当轮有后续任务依赖该结果
- 轮询必须在单次 bash 调用内完成：sleep 循环 + qstat，
  间隔 ≥60s，单次调用 ≤10 分钟，最多 2 次调用；超时即转手动模式，
  向用户报告"作业未结束，完成后请提醒我"
- 禁止跨 LLM 轮次的轮询循环（每轮都是 token 支出）

禁止：任何形式的"每 N 分钟回来看一眼"的 LLM 层轮询；
禁止在等待期间做与作业无关的推理消耗。

