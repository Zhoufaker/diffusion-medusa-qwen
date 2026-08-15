# D2 推理冒烟方案(必审档材料——随 T2 提交时一并报,未获批不上卡)

2026-08-15 定稿(用户任务 5)。前置:T1 结束、epoch-2 best 档产出。

## 形态

| 项 | 值 |
|---|---|
| 语料 | in-domain 300 的前 20 条(ordered,含 5 条 warmup 弃计) |
| 臂 | 三臂全跑(harness 原样,--n-prompts 20) |
| checkpoint | epoch-2 best 档(drafter_ckpt_full/drafter_best.pt) |
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
