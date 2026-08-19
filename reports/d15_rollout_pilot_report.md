# D1.5 rollout 长度 pilot 报告 v1

- 作业:176689374(dgxa100,walltime 10m27s,**实耗 12.54 SU**,exit 0)
- 代码:commit ee015b1(pbs/d15_rollout_pilot.pbs + scripts/prep_longform_pilot.py;
  gen_cache_rollout.py 零改动,Arm1 100% 复现谱系)
- 输入:longform_fixed_v2 池 37,079 条中 seed=43 分层抽样 50 条
  (docci 12 / detailcaps 10 / sp 16 / ln 12,配额与预注册规则见 pilot_manifest.json)
- 产出:50/50 rollout(fail=0),哨兵 JOB_DONE_176689374.json,样例 4 条

## 1. max_new 预注册裁定:512(机械触发)

预注册规则(pilot_manifest.json,数据到来前锁定):
> worst-source trunc@384 ≤5% → max_new=384;否则 512
> (512 为测量上限,超限如实登记仍取 512)

实测 worst-source trunc@384 = **0.600(DetailCaps)**,远超 5% 线 →
**裁定 max_new=512**。哨兵字段 `max_new_decision_prereg: 512` 由作业内
预注册逻辑自动写出,无人工干预。

### 如实登记:512 上限本身截断不轻

| 源 | n | mean | med | p90 | max | trunc@384 | trunc@512 |
|---|---|---|---|---|---|---|---|
| DOCCI | 12 | 346.9 | 306 | 512 | 512 | 41.7% | 33.3% |
| DetailCaps | 10 | 351.1 | 436 | 512 | 512 | 60.0% | 20.0% |
| SP | 16 | 363.9 | 378 | 512 | 512 | 43.8% | **12.5%** |
| LN-OI | 12 | 400.4 | 472 | 512 | 512 | 58.3% | **41.7%** |
| **pooled** | **50** | **366.0** | 392 | 512 | 512 | 50.0% | **26.0%** |

pooled trunc@512 = 26%:约四分之一的 rollout 在 512 处被测量上限截断,
真实自然长度分布右尾未知。预注册条款已预见此情形("超限如实登记仍取
512"),裁定不变;此数字作为已知测量局限入档。

叙事对照:ref 均长 v2 ~103(fixation report)曾引发"训练 token 量稀释"
预警;rollout 实测均长 366.0,方向证伪——question 模板恒定下模型输出
长度由图像内容驱动,与 ref 长度无耦合(v2 补记预判成立)。

## 2. T̄ 与特征缓存外推(本 pilot 最重要发现)

- 全序列均长 **T̄ = 1758.6**(prompt P_len + rollout L,n=50)
- 外推公式(哨兵内置):37,079 × T̄ × 5 层 × 3584 × 2B(fp16)
- **外推缓存体积 = 2,337.1 GB ≈ 2.34 TB**(对照旧线 ctx_cache_35k 仅 616G)

### 诊断:vision token 是体积主因(登录节点零 GPU 复算,n=50)

| 分量 | 均值 (tok) | 占 T̄ |
|---|---|---|
| vision token | **1,348.6** | **76.7%** |
| prompt 文本(模板+question) | 44.0 | 2.5% |
| rollout | 366.0 | 20.8% |
| **T̄** | **1,758.6** | 100% |

(prompt 整体 1,392.6 占 79.2%;复算 T̄ 与哨兵逐位一致)

per-source vision token 均值:DOCCI **3,864.9** / LN-OI 1,005.2 /
DetailCaps 551.6 / SP 217.1——v2 池图像分辨率差异极大,DOCCI 高清图
在无像素上限的 spec 下贡献了绝大部分序列长度。第一跑 pilot 沿用了
gen_cache_rollout.py 的默认 processor(无 max_pixels cap;旧线 34,999
缓存的图像为 COCO 小图,cap 从未 binding,故旧 spec 从未显式登记)。

### max_pixels=501760 投影(诊断复算,重跑实测为准)

- cap binding:25/50 张(DOCCI 侧为主);cap 后 n_vis 均 406.5、
  max 625(≤ 501760/28² = 640 理论上限,吻合)
- 若 rollout 长度 L 不变:T̄ → **~816.5**,缓存外推 → **~1,085 GB**
- 附注:L 在新 spec 下会变(图像下采样改变模型输出),以重跑实测为准;
  ~1TB 量级仍显著,是否需进一步措施待重跑后与规模/预算一并裁

## 3. 样例过目记录(每源 1 条,samples_for_review.txt)

| 样例 | 长度 | 收尾 | 文面检查 |
|---|---|---|---|
| docci_train_05478 | 252 | 自然收尾(im_end) | 散文三段,连贯,无退化 |
| detailcaps_WP-…_18-19153 | 512 | **上限截断**(Uncertainty 节句中断) | 五段结构化,截断前连贯 |
| sp_vg_2407118 | 498 | 自然收尾(im_end) | 五段结构化+总结段,连贯 |
| ln_oi_9fca31ec… | 472 | 自然收尾(im_end) | 四段结构化+总结段,连贯 |

- 4/4 无循环/复读/乱码等退化模式;无图对照下未发现明显幻觉标志
  (物体-场景关系自洽)
- 3/4 呈现同一五段结构化模板(Main Objects / Scene Context /
  Relationships / Visual Attributes / Uncertainty)——LONGFORM_PROMPT
  恒定模板下的模型风格趋同,属预期;docci 样例为纯散文,说明非强制
- detailcaps 样例的 512 截断即 §1 trunc@512 的实物示例

## 4. SU 台账

pilot 12.54 SU(预算 <15,dgxa100 10m27s,GPU util 26%,显存 18.9G)。
D1.5 线累计:copyq 首跑 77.19 + 纠正 5.72 + pilot 12.54 = **95.45 SU**。

## 5. 后续(裁决与重跑,见 v1.1 追記)

2.34TB 外推触发像素上限裁决(用户 2026-08-20):longform 线
PREPROCESS_SPEC 采用 **max_pixels=501760**(rollout 生成/特征抽取/
longform 评估三处同 spec,与 MM-Vet OOD 锚定约定对齐);短线旧 spec
不动,双 spec 并存,登记见 CLAUDE.md 与 d15_fixation_report.md 登记节。
pilot 按新 spec 同 50 条同 seed 重跑(免审档:同模板同预算,仅 prep
适配层加像素上限),max_new 按原预注册规则对新数据重裁。
