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
- 训练数据：34,999 条 self-distillation 缓存（Qwen2.5-VL-7B greedy rollouts，TARGET convention），与评估图像双键去重
- 计时锚点：pooled greedy 29.918 tok/s（旧线数值；新线 verify 形态不同，计时锚点需重新预注册后才可引用）

## 对比 baseline 数值（旧线冠军，只引用不重测）
- static 速度冠军：1.689×
- 动态树 dyn_k8_n32：σ=2.841，paired Δ+0.101±0.007
- MM-Vet OOD realization rate：R≈0.433±0.039

## 工作纪律
- 预注册优先：所有阈值、统计检验在数据到来前锁定
- 单变量消融；训练算力花费前必须过代码审查
- 任何对导师代码包内文件的操作一律先复制到本项目目录，原包只读

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
- scratch 两个已知风险，写入前评估：
  1. 100 天未访问自动清除——长期资产（如 base_lm_head、固定数据）需在报告中标注
     "建议本地备份"；训练期间高频访问的文件不受影响
  2. 项目级 inode 限额——大量小文件（如逐张图片、逐条 feature 文件）曾导致
     inode 超限、作业被 Held。批量小文件优先打包为单文件格式
     （tar / webdataset / 单个 safetensors 分片）
- 大体积写入（>50G）前先跑 lquota 报告余量，写入后再报告一次
- 登录节点默认无 Python 环境，任何 python 命令前先 module load + activate（见环境节）

