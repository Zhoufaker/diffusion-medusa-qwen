# D2 评估协议(预注册·锁定版)

2026-08-15 定稿。锁定后任何改动需用户明示,并在文末修订历史留痕。

## 范围与语料
- in-domain 300-prompt 嵌套 manifest(seed=43),旧 100 条为字节级回归门
- max_new_tokens=256,greedy(T=0)
- 硬件:A100 dgxa100 单卡;三臂统一 attn_implementation="sdpa" 显式锁定
- OOD(MM-Vet 218,V100+sdpa)属 D4,不在本协议

## 三臂与计时
臂① AR greedy(增量解码,旧线 clean greedy 同代码路径)
臂② dyn_k8_n24 复跑(旧线速度冠军)
臂③ diffusion drafter(block 16,best checkpoint)
同作业同进程交替;prompt 内三臂 Latin square 轮转;前 5 条 warmup 弃计;
计时点 cuda.synchronize 包夹;逐 prompt JSON 落盘。

## 门
**G0a 代码保真(硬门,V100)**:
- 在 **V100 gpuvolta** 上以旧线回归门入口**原样**复跑臂② 配置于 old-100,
  与 v2 参照 archive(v2_rebaseline/dyn_k8_n24.spec_archive.json,同为
  V100 录制)**字节级一致为硬门,失配即判负**
- 免审档:已过审模板重跑(仅改输入清单/作业名),预估 ~5 SU;
  可先于 A100 主评估独立执行
**G0b 计时基准(A100)**:
- G0a 通过后,A100 harness 臂② 复跑加速比值**即新基准**(硬件迁移,
  无数值对照门);保真依据 = G0a 字节门 + 冠军 flags 逐项对照
  (job 175598529 日志头)
- σ_batch = 0.0537 仅登记(2σ = 0.1074,带 [1.6246, 1.8394]。
  出处与计算:e2e_speed_300/dyn_k8_n24.json 的逐 prompt paired_speedup
  (job 175598529,300 条,提交顺序),按 25 条/批 × 12 批取批均值后的
  样本标准差;批均值序列 [1.6084, 1.766, 1.6929, 1.7638, 1.7159, 1.7476,
  1.7741, 1.8227, 1.7278, 1.7552, 1.6945, 1.7149])
- 锁定时判定依据(保留):ARCHIVE_POLICY_V2_ANCHOR.json 无硬件字段;
  实查 v2 计时作业 175598529 与 175680071/72 均为 `#PBS -q gpuvolta`
  (12 核/96G 单 V100 形态,PBS 记账佐证)→ v2 计时硬件为 V100
**G1(唯一硬门)**:臂③ 加速比 > 臂②,paired 逐 prompt 差值的
bootstrap 95% CI 不跨零。
- **加速比定义(v1.2 补全)**:speedup_X = (wall_①/n_①)/(wall_X/n_X),
  三臂同式;n 为该臂实际生成 token 数(截至各自 EOS/max_new)。
  即 per-token 时间之比,消除跨臂输出长度差异(漂移/EOS 不等)对
  纯壁钟比的混淆
- **计时口径登记**:三臂均为端到端壁钟(含 CPU 图像预处理),
  与 v2 e2e paired 口径一致
- CI 跨零且点估计更高 → 扩至 3 次重复计时
- 扩后仍跨零但点估计仍高:记"G1 边缘",带完整分布与导师共同裁决,
  不自动判负、不自动放行
**G2 确定性**:臂③ 同配置双跑,输出逐 token 一致。
**τ 只登记不设门**:两口径(纯接受 draft 数 / 接受+bonus)定义现场写死,
不与旧线 σ 混算;对照 DFlash τ=6.5 时必须带 35K vs 800K 数据折扣脚注。

## greedy 等价性声明
臂③ 输出为分块 teacher-forcing 前向下的 greedy,与臂① 增量 greedy 之间
存在已量化的前向形态数值漂移(~3.9%/位置,四臂归因矩阵,非 bug)。
本协议声明:**两者视为等价的 greedy 实现**,速度对比以各自输出为准,
不要求跨臂逐 token 一致;臂③ 的 lossless 定义为
"lossless w.r.t. teacher-forcing verification"。
登记项:臂③ vs 臂① 逐 token 精确匹配率如实报告;抽 10 条分叉样本
人工过目,确认分叉点为漂移型(实词选择位)、分叉后无质量退化。
本声明于本周与导师同步。

## D3 埋点(随臂③ 免费产出)
逐 cycle 落盘:每槽位 drafter top-1 概率、accept/reject、
reject 处 verify argmax → 块内位置接受率曲线 + per-slot 置信度分布。

## off-policy 附属实验(随 D2 一并出,~10 SU)
val-500 上 drafter 单遍前向,按"缓存 token == A100-TF argmax"分层
(≈96% / 4%),统计两层的 drafter top-1 命中 verify-argmax 率。
预注册方向性预期:4% 层命中率显著低于 96% 层。
产出:off-policy 对整体接受率损失的上界估计 ≈ 4% × 两层命中率差。

## 报告清单
G0-G2 判定、τ 两口径、跨臂匹配率与分叉抽检记录、D3 埋点数据路径、
off-policy 附属实验、SU 实耗 vs 预算、逐 prompt 原始 JSON 路径。

---

## 修订历史
- v1.0(2026-08-15):定稿锁定。锁定时填入两处 G0 写死项:
  ① σ_batch=0.0537 及计算出处;② v2 计时硬件实查为 V100(gpuvolta)
  → 适用"原为 V100"分支,无数值对照门。另附 old-100 字节门 V100 参照
  archive 的硬件漂移登记条款(报告义务,非门变更)。
- v1.1(2026-08-15,用户明示授权):删除 v1.0 附加的"登记性条款"
  (old-100 A100 失配报告裁决制)——该条款把硬门软化为登记制,且根因是
  参照档硬件(V100)与复跑硬件(A100)不一致。改为分硬件双段:
  **G0a 代码保真**(V100 同硬件字节级硬门,失配即判负)+
  **G0b 计时基准**(A100 复跑值即新基准,保真依据=G0a+冠军 flags 对照)。
- v1.2(2026-08-15,用户已批):G1 加速比定义补全为 per-token 式
  speedup_X = (wall_①/n_①)/(wall_X/n_X),并登记端到端壁钟计时口径
  (含 CPU 图像预处理,与 v2 e2e paired 口径一致)。
  原因:跨臂输出长度因漂移/EOS 不等,纯壁钟比混淆吞吐与响应长度。
