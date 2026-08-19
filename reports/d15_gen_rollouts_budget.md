# D1.5 v2 批量 rollout 生成 — 送审件(必审)

状态:**草案**。吞吐槽位(标 TBD)待 px 重跑 pilot(job 176694274)哨兵
出数后回填,回填后整包送审;未获批不 qsub。

## 送审对象
1. 代码:scripts/gen_longform_rollouts.py(新脚本首次上卡→必审)
2. 单测:tests/test_gen_longform_rollouts.py(CPU,**19/19 过**:
   左 pad 正确性/续跑幂等/record+manifest schema/EOS 口径/连续分片)
3. PBS 草案:pbs/d15_gen_rollouts.pbs(dgxa100 array 0-7,walltime TBD)

## 任务与产物
- 输入:longform_fixed_v2 全池 37,079 条(triplets.jsonl + 38 个图像 tar)
- 输出:/scratch/…/longform_fixed_v2/rollouts_px501760/
  逐 shard JSONL ×8 + manifest.json + 逐子作业哨兵
- 体积/inode 账:token JSONL 总计 ~0.1G、新增 inode ~18 个(8 数据文件+
  哨兵+manifest)——对 616G 特征缓存问题无贡献,inode 警戒不受扰
- **不产特征**:hidden 由队列 ②(W1 extract_ctx_features 模板适配)另跑;
  2.34TB→~1.1TB 体积问题在 ② 的预算单裁,与本单解耦

## 谱系口径(审查重点,均已写入脚本 docstring 与 manifest config)
| 项 | bs=1 谱系(gen_cache_rollout) | 本脚本批量路径 |
|---|---|---|
| 解码 | 裸 argmax(argmax_masked) | do_sample=False + repetition_penalty=1.0 显式压回(generation_config 自带 1.05,不压即分叉) |
| EOS | 只停 tokenizer.eos(151645),首 EOS 计入序列 | eos_token_id 显式单值传入,trim 首 EOS 计入;无 EOS 达 max_new → eos_hit=false 登记 |
| 幻影词表 | mask_phantom_(ids≥151936→-inf) | 等价 LogitsProcessor,EFFECTIVE_VOCAB 与 decode.common 断言一致(单测覆盖) |
| 像素 | (pilot 重跑经 prep 预缩放) | apply_max_pixels(processor, 501760)——PREPROCESS_SPEC 正典路径 |
| 数值 | — | **artifact 锁定制**:锁逐条 token+sha256,批内 fp16 扰动属已知类别,登记不视为缺陷 |

## 吞吐外推预算(公式锁定,数字 TBD)
- R₁ = px 重跑 pilot bs=1 实测吞吐 = **TBD** row/s
  (对照:首跑无 cap 0.093 row/s,GPU util 26%,预期 cap 后显著上升)
- 批量效率假设:bs=8 相对 bs=1 加速 E ∈ [3, 6](util 26% 有大余量;
  取保守下界 E=3 做预算,E<1.5 即中止改预案)
- GPU·h = 37079 / (R₁ × E × 3600);SU = GPU·h × 72(dgxa100)
- 每 shard walltime = (37079/8) / (R₁ × E) + 10min 加载余量,
  ×1.5 排队安全系数后填入 PBS
- **预算数字(回填)**:GPU·h = TBD;SU = TBD(E=3 保守值)/
  TBD(E=6 乐观值);walltime/shard = TBD

## 验证方案(获批后执行顺序)
1. 冒烟(必审单内含):--limit 64 --num-shards 1,独立 out-dir,
   验 JSONL/哨兵/manifest 全链 + 实测 bs=8 有效吞吐(回填 E)
2. 断点演练:冒烟中途 qdel 一次,重提验幂等(文件级 sha 对照)
3. 全量 8-shard array
4. 事后对照(非门,登记):50 条 pilot 行的批量输出 vs bs=1 pilot 输出,
   报 token 级一致率——fp16 批扰动的实测标定,写入生成报告

## 免审/必审判定
必审(新脚本首次上卡 + 写入新位置);预估 SU 待回填(按首跑 pilot 比例
预计 < 200,但以回填数为准);无删除/移动操作;不触碰锁定资产
(triplets.jsonl 与图像 tar 只读)。
