# D1.5 v2 批量 rollout 生成 — 送审件(必审)

状态:**已回填,整包送审中**(吞吐锚 = px 重跑 pilot job 176694274 实测,
2026-08-20 回填);未获批不 qsub。

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

## 吞吐外推预算(已回填,锚 = job 176694274)
- R₁ = px 重跑 pilot bs=1 实测吞吐 = **0.0958 row/s**(50 条/8.7 min,
  dgxa100,GPU util 42%)。附注:对首跑无 cap 0.093 仅 +3%——bs=1 下
  瓶颈在逐 token decode(rollout 均长 371.9 基本未变),cap 省的是
  prefill;故批量化(decode 并行)才是吞吐主杠杆,E 假设区间维持
- 批量效率假设:bs=8 相对 bs=1 加速 E ∈ [3, 6](util 42% 仍有余量;
  取保守下界 E=3 做预算,冒烟实测 E<1.5 即中止改预案)
- GPU·h = 37079 / (R₁ × E × 3600);SU = GPU·h × 72(dgxa100)
- **预算数字**:
  - E=3(保守,预算基准):**35.8 GPU·h → 2,581 SU**
  - E=6(乐观):17.9 GPU·h → 1,290 SU
  - walltime/shard:4,635 行 / (0.0958×3) = 4.48h + 0.2h 加载,
    ×1.5 安全系数 → **07:00:00** 已填入 PBS
  - 绝对上限登记:若 8 shard 全数跑满墙钟,8×7h×72 = 4,032 SU
    (PBS 按实际走时计费,预期落在 1,300–2,600 区间)
  - 冒烟(获批后先跑):对**同一 64 行**(--limit 64,池首 64 行)
    同作业内先 bs=8 后 bs=1 各跑一遍(独立 out-dir),
    **E = R_bs8 / R_bs1 同条配对**——不用 R₁(pilot 分层 50 条)外推,
    避免行集构成差异混进 E(冒烟行全 DOCCI);walltime 00:40:00,
    预估 ~25-30 SU(bs=1 臂 ~11 min 为配对成本),实测 E 后按上表
    对号入座;全量预算仍用 R₁×E(R₁ 代表全池分层构成)

## 验证方案(获批后执行顺序)
1. 冒烟(必审单内含):--limit 64 --num-shards 1,独立 out-dir,
   验 JSONL/哨兵/manifest 全链 + 同条配对实测 E(bs=8 vs bs=1,
   口径见上节;E<1.5 中止改预案)
2. 断点演练:冒烟中途 qdel 一次,重提验幂等(文件级 sha 对照)
3. 全量 8-shard array
4. 事后对照(非门,登记):50 条 pilot 行的批量输出 vs bs=1 pilot 输出,
   报 token 级一致率——fp16 批扰动的实测标定,写入生成报告

## 免审/必审判定
必审(新脚本首次上卡 + 写入新位置 + 预估 SU 2,581 > 500 三重触发);
无删除/移动操作;不触碰锁定资产(triplets.jsonl 与图像 tar 只读)。
大写入前 lquota 已查(2026-08-20):scratch 5.05/10 TiB、inode
732K/1,018K——本单产物 ~0.1G/18 inode 无压力;~1.09TB 属队列 ② 议题。
