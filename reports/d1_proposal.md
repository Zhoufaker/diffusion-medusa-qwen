# D1 复现方案（一页纸，方案待确认）

**目标**：在 Qwen2.5-VL-7B-Instruct 上复现 DFlash 式 block diffusion drafter
（arXiv:2602.06036），核心研究问题：接受率的校准。
数字全部引自 `dflash_survey.md`（2026-08-12，实测推算）。

## 现状与关键事实

- 官方仓库（z-lab/dflash，commit 94e4abc）**仅开源推理**（1,492 行）；训练配方未开源，
  需按论文自研——这是本方案最大的工程量与最大风险
- 兼容性实测通过：drafter 代码在我们的 transformers 5.3.0 + torch 2.5.1 环境**原样可导入**，
  无需降级；V100 上推理可走 sdpa 回退，**OOD gate 可继续锚定 V100**（预注册写死 sdpa）
- 架构即插即用度高：drafter 复用 target 的 embed/lm_head（Qwen2.5-VL-7B 两者非 tied，
  照常各自引用），可训练参数 ≈ **1.23B**（5 层 + 融合 fc）；drafter 用一维 RoPE，
  不需引入 M-RoPE
- 旧线 34,999 缓存的 hidden **不可复用**（只有最终层、只有 rollout 段）；但 rollout
  轨迹与数据谱系完整保留，只需对固定 (prompt, rollout) 重抽 5 个中间层
  [1,7,13,19,25]——三层评估体系不动

## 推荐路线

**离线 5 层 feature 缓存（含 vision token 的完整上下文，方案 A）+ 分片打包**：

| 项 | 数值 | 依据 |
|---|---|---|
| 缓存体量 | ≈ 658 GB（34,999 条 × 均长 525 × 5 层 × 3584 × bf16） | 长度为 400+200 条实测抽样 |
| 存储可行性 | scratch 余 5.8 TiB ✅；inode 余仅 53K → 打成 ~137 个 safetensors 分片 | 本日 lquota |
| 抽取算力 | 2–4 A100·h（一次性） | target 单次 forward/条 |
| 训练算力 | 单次 8–27 A100·h（anchor = min(512, α×有效位置数)，α∈[1,2]，按 rollout 均长 144 估 150–300/序列；**最终值 W2 代码审查时锁定**） | 6 epochs，FLOPs 3.7–7.4e18 |
| SU 费率 | **实测 72.0 SU/GPU·h**（dgxa100，PBS 记账反推，两作业吻合） | survey §B5c |
| 总预算 | **约 4–8 KSU**（含 3 次重训 + 抽取 + 评估；anchor 取满 512 的保守上限 ~13 KSU） | 实测费率重算 |
| SU 余额 | 137.73 KSU（q3）→ 预算占比 3–9% | 本日 nci_account |

选离线而非在线抽取的理由：特征确定性可复现（契合预注册/exactness 纪律）、
训练显存与管线最简、在线路线会把图像预处理成本重复付 6 遍。
vision token 进入 context feature（数据 100% 是图像任务，不预先砍视觉通道；
剔除 vision 的对照可作后续单变量消融）。

## 风险

1. **训练配方自研**（anchor 采样 / 块内 mask / loss decay 全部从论文复原）——官方
   "recipe coming soon"，建议 watch 仓库，官方放出后立即对表校正
2. transformers 5.3.0 与仓库 pin 的 4.57.1 的运行期行为差异（导入已通过，行为待 smoke）
3. `logits_to_keep` / `output_hidden_states` 在 Qwen2.5-VL forward 上的兼容性待 smoke
4. 接受率校准（研究主线）本身在 VLM 上无先例，视觉段接受率分布未知
5. inode 水位紧（余 53K）：一切新产物强制分片打包，>50G 写入前后跑 lquota
6. **训练量级差距**：官方 800K 样本 vs 我们 35K（~1/23）——D1 gate 阈值预注册
   按小数据折扣设定，不锚论文 τ=6.5；SU 预算按实测费率后不再是忽略量级，
   重训须过审查（W2 参考：dspark-aeon-27b 社区 recipe、DFlare arXiv:2606.02091）

## 时间线（预计 4 周，方案待确认）

- **W1**：抽取脚本 + 5 层缓存全量生成 + exactness 审计（对旧缓存 tokens 逐条一致性校验）
- **W2**：训练循环自研（anchor/mask/decay）+ 单元测试 + **代码审查门**（算力花费前）
- **W3**:首次训练（dgxa100）+ 训练分布/in-domain 300 评估 + 接受率校准初判
- **W4**：MM-Vet OOD（V100, sdpa, max_pixels=501760）+ 计时锚点重新预注册 + D2 方案

> 本方案为 D0 调研产出，所有路线选择（方案 A、离线缓存、V100 锚定）**待导师确认后执行**。
