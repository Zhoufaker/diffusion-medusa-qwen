# W2 smoke 训练报告 — 六门逐项 + 全量外推

- Job：176278959.gadi-pbs（dgxa100 1×A100），exit 0
- walltime 22:33（申请 1:30:00），**实耗 27.06 SU**
- 口径（F2）：train = 前 2000 条非 val 索引 × 1 epoch（500 步，batch_seqs=4）；
  val = seed43 固定 500 条（`val_indices.json` 已落盘）；`--self-check` 开启
- 产物：`/scratch/li96/mz9869/dflash_data/drafter_ckpt_smoke/`
  （drafter_best.pt / drafter_step500.pt 各 ~2.5G、train_manifest.json、
  epoch_0_stats.json、train_summary.json、哨兵 JSON）
- **结论先行：六门 5 过 1 边缘（门 3 显存偏高但在上界口径内），无失败项；
  训练循环端到端跑通，loss 下降正常，全量训练单次外推 ~30–32 GPU·h ≈ 2.2–2.3 KSU。**

## 六门逐项

| 门 | 判据 | 实测 | 判定 |
|---|---|---|---|
| 门1 | 500 步 weighted CE 趋势线斜率为负 | 全 10 点最小二乘斜率 **−0.0071/步**；剔除 step0 暖启动点后 **−0.0020/步**，仍显著为负 | ✅ |
| 门2 | 首 batch fp32 sdpa-vs-eager max\|Δ\|<1e-4 + 形状/有限性 | **3.14e-05**；hidden 形状与有限性断言通过（self-check 内嵌） | ✅ |
| 门3 | 显存峰值 vs 预估 28–30G 偏差 <30% | torch 峰值 **37.87 GiB**：对上界 30G **+26.2%（过线）**，对下界 28G +35.2%（超线）；PBS 口径 GPU Memory Used 51.16GB（含 CUDA context/allocator 缓存） | ⚠️ 边缘 |
| 门4 | best 档 reload 后 val CE 复现差 <1e-3 | **1.44e-4**（6.569338 → 6.569482，bf16 存档量化损失） | ✅ |
| 门5 | off-policy 聚合 vs manifest 直算一致（含 decile 断言） | share=0.042056 双算一致，decile 直方图断言通过 | ✅ |
| 门6 | 哨兵 + 邮件到位 | 哨兵含 best_val_ce/peak_gpu_mem_gib/gate4 字段；邮件已送达（你据此回来） | ✅ |

### 门 3 边缘的归因假设与行动项

超出预估的 ~8G 大概率来自：① sdpa 带任意 4D mask 时走 math/mem-efficient 后端，
注意力矩阵 (28 头×~3.6K×~4.1K) 有实体化开销（预估时按 flash 路径想当然）；
② self-check 的 fp32 前向（参数临时 ×2）与训练峰值同进程叠加；
③ lm_head logits (13.8K×152064) 的 fp32 CE 工作区大于分块预估。
80G 卡余量仍充足（47%），不阻塞全量；行动项：全量训练时 `batch_seqs` 保持 4
（不升 8），显存实测入全量报告；W3 FlexAttention 消融连带解决 ①。

## loss 曲线数据点（epoch 0，每 50 步）

| step | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 |
|---|---|---|---|---|---|---|---|---|---|---|
| weighted CE | 12.022 | 7.061 | 7.237 | 7.226 | 7.192 | 6.684 | 6.850 | 6.572 | 6.477 | 6.546 |

- step0 = 12.02 ≈ ln(151936)=11.93（随机初始化的均匀分布水平）——初始化健康
- epoch 末：train 6.890 / **val 6.569**（困惑度 ~713；2000 样本 × 1 epoch 的
  冒烟量级，不代表收敛水平）
- off-policy 监控（§1f 埋点工作正常）：train 4.171% / val 4.206%，与全量漂移
  登记 4.137%（=1−95.863%）一致；head16 截断样本 54/2000（覆盖率 97.3%）

## 全量训练外推（按实测 72.0 SU/GPU·h）

- 实测步速：~2.0–2.1 s/步（batch 4 序列 ≈ 13.8K noise token/步；MFU ~13%——
  偏低，主因 batch 小 + sdpa math 路径 + fp32 参数 autocast，W3 优化空间）
- 全量：train 34,499 条（35k − 500 val − 0 过滤*）→ 8,625 步/epoch × 6 epochs
  = 51,750 步 → **~30–32 GPU·h**（含 val/ckpt 开销 ~6%）
- **SU：单次全量 ≈ 2,200–2,330 SU**（保守含队列波动上限 ~2.7 KSU）
- 对照：survey 早先乐观带 0.6–2.0 KSU 需上修至 ~2.2–2.7 KSU/次；
  含 3 次重训的 W2 总预算 ~7–9 KSU，仍在余额（137.73 KSU@q3）的 ~6% 内
- walltime 申请建议：单档 34h 超 dgxa100 常规上限时分 epoch 续跑
  （`--init-from` 已就位，正是为此），如 2 epoch/档 × 3 档，每档 ~11h
- *全量 manifest 无 L<2 样本（`train_manifest.json` n_filtered_short=0）

## 状态与待你决策

- **全量训练 = 必审档**：如批准，我发正式预算 + PBS（分档续跑方案）过审后提交
- W1 全量报告（含四臂两两矩阵最后一格）在你上次断线前被中断，等你指示再补
- 旧缓存收编（onpolicy_data 散件 ~30.6K inode + llava_general_35k tar）继续等单独授权
- git 干净（`4c71fad`）；无在途作业
