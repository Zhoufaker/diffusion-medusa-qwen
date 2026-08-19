# d15_extract_budget — 队列②:longform 特征抽取预算单(必审)

- 对象:37,079 条 longform rollout 的 5 层 teacher-forcing 特征抽取,
  产出训练缓存 dflash_data/longform_ctx_cache/(独立于 ctx_cache_35k)
- 必审触发:新脚本(scripts/extract_ctx_longform.py)首次上卡 +
  写入新位置 + TB 级大体积写入
- **前置依赖:队列① 全量收账 complete=true 后才可上卡**(适配层
  load_locked_rollouts 亦从代码上强制:manifest 非 complete 拒读)
- 状态:**备审。未获批不 qsub。**

## 1. 代码与单测(包内)

- scripts/extract_ctx_longform.py:W1 extract_ctx_features.py 适配层,
  改动仅限四处(输入源+锁定先验 / 像素 spec 501760 / 生成端逐参数镜像
  的 TF 输入构造 / 分片写 longform_ctx_cache),其余零改动:
  层 [1,7,13,19,25]、前向 fp16 sdpa、存储 bf16、(5,T,3584)、spans 六元组、
  SHARD_SIZE=256、safetensors key schema。TF argmax 诊断登记不设门
  (前向形态差异已知类别,W1 旧缓存门不移植——见脚本 docstring)。
- 单测 6 条已过(CPU,登录节点):三级先验拒坏(complete=false/文件
  sha/行级 validate_record)、spans 构造(longform 无多轮:prompt 段=
  模板+图+问题,rollout 段=锁定序列;缺图即停)、分片 round-trip
  (bf16 保真)、层位点对 W1 一致。

## 2. 体积预算

- 公式:bytes ≈ N × T̄ × 5 × 3584 × 2(bf16;ids/spans/manifest 忽略不计)
- 预期(pilot v1.1 实测 T̄=822.4,px501760 新 spec):
  37,079 × 822.4 × 35,840 B ≈ **1,093 GB ≈ 1.02 TiB**;
  145 片(ceil(37079/256)),均 ~7.5 GB/片;**inode 增量 ~146**
- 【收账后回填】精确数:队列① 报告出全池 L̄(分源)后,按
  T̄ = P̄(pilot 实测像素分布)+ L̄(收账实测)回算,±5% 内不改预算,
  超出重报
- 分片器物理约束:单片 256 条 × ~30 MB ≈ 7.5 GB,dgxa100 内存/写盘无虞;
  批量小文件纪律满足(145 个大文件,非散件)

## 3. 算力预算

- 吞吐锚:W1 全量实测 **1,921 tok/s**(A100 fp16 sdpa、同层位点、同
  整序列 TF 前向、同 bf16 落盘路径——形态完全同款,直接可用)
- 总 token:37,079 × 822.4 ≈ 30.5M tok
- 全量:30.5M ÷ 1,921 ≈ 15,870 s ≈ **4.41 GPU·h ≈ 318 SU**(dgxa100 72/h)
- 执行形态:**单卡单作业**(4.4h < 走 7h walltime,含 1.5 安全系数;
  不分 shard 作业,一份 manifest,免合并)
- 先行 pilot:--end-idx 512(2 整片,~7 min 生成 + 模型加载),walltime
  00:30,**~12 SU**——验 safetensors 格式/spans/体积外推(512 条实测
  GB 数 ×72.4 外推全量)/TF posmatch 诊断落点,过目后放全量
- **预算数字:预期 ~330 SU(pilot 12 + 全量 318);绝对上限 516 SU
  (pilot 36 + 全量墙钟 7h×72=504,按实际走时计费)**

## 4. 存储现状与写入后预测(2026-08-20 实测 lquota)

- scratch:**5.05 TiB / 10.00 TiB**(硬限 10.50);写入 +1.02 TiB 后
  预计 **~6.1 TiB / 10.00 TiB**(61%)
- inode:**732,438 / 1,018,000**(软限);+146 后 ~732,600,无虞
- 写入前后各跑一次 lquota 报告(>50G 纪律)
- 大额写入并存现状:ctx_cache_35k 616 GB(旧线训练缓存,10 月下旬
  清除线)+ 本次 1.02 TiB。**项目盘提示(建议发导师,草稿)**:
  > 李老师:新线(block diffusion drafter)训练特征缓存本周落盘,
  > 预计新增约 1.1 TB 至 /scratch/li96(项目盘现用 5.05/10 TiB,
  > 写入后约 6.1/10 TiB)。旧线 616 GB 特征缓存在新线训练跑通后可转
  > 冷备腾挪;如项目组近期另有大用量计划请提前告知,我可调整落盘节奏。
- atime 提示:训练期高频访问不受 100 天清除影响;训练间歇期将
  longform_ctx_cache 加入 touch 名单(CLAUDE.md 已有机制)

## 5. 谱系与先验(审查要点)

- 锁定制下游第一读者:load_locked_rollouts 三级先验(manifest
  complete → 逐文件 sha256 → 逐行 validate_record),任一不符即停;
  triplets.jsonl 复用 verify_pool(37,079 + sha pin 06b6f55b…,默认值
  即断言,无需 PBS 显式传参)
- 抽取 manifest 记录:preprocess_spec(runtime 断言 max_pixels==501760)
  + source_lineage(rollout manifest sha256 / rollout code_commit /
  triplets sha / images tar 目录)+ code_commit + TF 诊断聚合

## 6. 执行顺序(获批后)

1. pilot 512(独立 out-dir …/longform_ctx_cache_pilot,免污染正式位)
2. pilot 过目(格式/体积外推/诊断)→ 回填第 2 节精确数
3. 全量单作业(正式 out-dir,walltime 07:00)
4. 收尾:manifest 过目、lquota 复报、touch 名单登记、交接文档
