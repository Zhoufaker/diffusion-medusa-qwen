# D2 推理冒烟方案(必审档材料——未获批不上卡)

2026-08-15 定稿,2026-08-16 随 T2 提交更新。前置已满足:T1 结束(exit 0,
val 3.962→3.584 连降,中止判据未触发)、G0a 已过(100/100)。

## 形态

| 项 | 值 |
|---|---|
| 语料 | in-domain 300 的前 20 条(ordered,含 5 条 warmup 弃计) |
| 臂 | 三臂全跑(harness 原样,--n-prompts 20) |
| checkpoint | **drafter_best_ep2_pinned.pt**(epoch-2 best 的固定副本——T2 运行中会覆写 drafter_best.pt,冒烟引用不动的 pinned 档,与 T2 无文件冲突) |
| 硬件 | A100 dgxa100 1 卡,sdpa |
| walltime | 1h |
| 预估 SU | **<30**(20 prompts × 三臂 ≈ 分钟级 + 模型加载) |
| G0a 依赖 | --g0a-result 指向 176356106 产出(须已 pass) |

## 冒烟检查项(除 harness 自动判定外)

1. **G2 双跑**:harness 内嵌 3 条(自动)
2. **臂③ 输出人工过目 3 条**:重点核对 rope_deltas 整块 verify 的合理性
   (无重复段/乱码/位置错乱——M-RoPE 多 token 续接路径的目检)
3. **臂③ vs 臂① 匹配率**:落在漂移预期量级(~96%/位置附近);
   仅登记不设门(协议 greedy 等价性声明)
4. 吞吐/显存记录,为 300 条正式跑的 walltime 外推提供依据

## 产出

- d2_smoke_summary.json(harness 汇总)+ 3 条人工过目记录
- 判定:全绿 → 申请 300 条正式 D2;异常 → 带 D3 埋点数据分析报告
