# W2 全量训练正式预算包（必审档，等待批准）

2026-08-15。依据：smoke 176278959 实测（2.0–2.1 s/步，27.06 SU/22.5min，
峰值显存 37.87 GiB）；费率实测 72.0 SU/GPU·h（dgxa100）。

## 1. 队列上限与分档形态

- `qstat -Qf dgxa100(-exec)` 未暴露 resources_max.walltime（路由队列不可见）；
  可用证据：本项目历史训练作业 `c1_train.pbs` 以 **walltime=14:00:00 在 dgxa100
  过审并完成**；NCI 文档口径上限 48h
- 单档全量（~31h）名义上放得下 48h，但：①长 walltime 排队/backfill 显著变差；
  ②31h 单点故障成本高。**选择：3 档 × 2 epochs**，档形状 14h（历史验证过的
  提交形态），bundle 既是档间续跑机制也是档内容错
- resume 等价性已验证（免审档、零 GPU）：CPU 小模型 1+resume+1 vs 连续 2 epoch，
  **逐步 loss 比特级一致（容差 0）**，含 optimizer/scheduler/RNG 状态与
  (seed, epoch) 派生的 shuffle/anchor 顺序（tests/test_w2_drafter.py::
  test_resume_bundle_two_epoch_equivalence，16/16 全绿）

## 2. 分档表

| 档 | epochs | 预计计算 | walltime 申请 | 预计 SU |
|---|---|---|---|---|
| T1（RESUME 空，UNTIL=2） | 0–1 | ~10.5–11h | 14h | ~760–800 |
| T2（resume bundle，UNTIL=4） | 2–3 | ~10.5–11h | 14h | ~760–800 |
| T3（resume bundle，UNTIL=6） | 4–5 | ~10.5–11h + 门4 复验 | 14h | ~780–820 |
| **合计（单次全量）** | 6 | ~31–33 GPU·h | — | **~2,300–2,420 SU** |

- **3 次重训上限：~7.3 KSU**（占 q3 余额 137.73 KSU 的 ~5.3%）
- 排队策略：**档间人工门**——T(n) 结束后先查 progress.json 的中止判据
  （§4），通过才提交 T(n+1)；不用 PBS depend 链（中止判据需要人在环）
- 步速依据：smoke 中段 106s/50 步 = 2.12s/步；全量 8,625 步/epoch
  （34,499 条 ÷ batch 4）；显存按 smoke 实测 37.87G，batch_seqs 维持 4 不上调

## 3. checkpoint 存储预算（scratch）

| 项 | 体积 |
|---|---|
| bundle_latest.pt（仅最新一份，写后删旧） | ~15 G |
| 6 个 epoch 档（drafter bf16） | 6 × 2.5 G = 15 G |
| drafter_best.pt | 2.5 G |
| **合计** | **~32.5 G**（scratch 余 ~5.4T，无压力；inode +9） |

## 4. 监控与中止判据（已写入 PBS 模板头注释）

- 哨兵 JSON 每档写：`epochs_done`、`best_val_ce`、`last_val_ce`、`bundle` 路径、
  本档 val CE 历史（源自 progress.json，训练循环每 epoch 落盘）
- **中止判据**：val weighted CE **连续 2 epoch 上升** → 停止后续档提交，
  带曲线报告用户裁决（**不自动 qdel 运行中的档**——档内 2 epoch 自然收尾，
  bundle 保留现场）
- off-policy 聚合每 epoch 照旧输出（epoch_*_stats.json，§1f 埋点）
- 门 4（加载保真度 <1e-3）在 T3 完成全部 epoch 后自动执行

## 5. 训练数据快照（train_manifest.json 自动写入，已实现）

| 字段 | 值/来源 |
|---|---|
| cache_dir | /scratch/li96/mz9869/dflash_data/ctx_cache_35k |
| cache_manifest_created / code_commit / hardware | 运行时从 ctx_manifest.json 读取（2026-08-15 / 4c71fada / A100-dgxa100） |
| train_indices_sha256 / val_indices_sha256 | 索引清单 JSON 序列化后 sha256，随 val_indices.json 落盘 |

## 6. 执行清单（批准后）

1. qsub T1（`UNTIL=2` 无 resume）—— 模板 pbs/w2_full_train.pbs 已过形状审查
   （资源行与 smoke 相同 + walltime 14h）
2. T1 完成 → 读哨兵查中止判据 → qsub T2（`UNTIL=4 RESUME=<bundle>`）
3. 同理 T3；完成后出全量训练报告（loss 曲线、门 4、off-policy 分段、
   walltime/SU 实耗 vs 本预算）
