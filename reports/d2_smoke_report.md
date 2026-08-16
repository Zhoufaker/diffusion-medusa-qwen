# D2 冒烟报告(异常分支——诊断版)

2026-08-16。Job 176390465(dgxa100),**exit 1,walltime 2m36s,实耗 3.12 SU**。
按方案异常分支执行:停、出诊断、不重试、不改代码。T2 未受影响。

## 故障定位(唯一根因,证据完整)

**sys.path 模块遮蔽**:`scripts/train.py`(旧线训练入口,模块)遮蔽了
`train/`(新线训练包)。

```
harness main() → sys.path.insert(0, _ROOT/"scripts")   # 为导入 eval_acceptance_tree
→ load_drafter_for_inference()                          # dflash_vlm.py:40 惰性 import
→ from train.train_drafter import ...                   # 解析到 scripts/train.py!
→ scripts/train.py:37: from train.trainer import Trainer
→ ModuleNotFoundError: 'train' is not a package
```

- 失败点在**drafter 加载之前**:base(56s)与旧线 head(88s)已正常载入,
  三臂零执行、零计时数据、D3 目录空(0 文件)
- 输出目录仅哨兵(exit=1 + summary 缺失注记),per_prompt 0 条

## 为何单测未拦截(诚实登记)

1. mock 全流程测试从不调用 `load_drafter_for_inference`(其内部 import 是
   函数级惰性的——恰是被遮蔽的那行)
2. W2 测试在同 pytest 进程中**先**导入了 `train` 包 → `sys.modules` 缓存
   掩护了后续遮蔽
3. 教训:路径操纵(insert(0))+ 同名模块/包共存 + 惰性 import 三者叠加,
   构成单测盲区

## 修复方案(待授权,一行级改动 + 回归测试)

1. `scripts/d2_eval_harness.py`:
   a. `sys.path.insert(0, scripts/)` 改为 **append**(train 包优先解析);
   b. 在插入 scripts/ 之前**先行 eager import** `train.train_drafter`
      (双保险,消除对 path 顺序的依赖)
2. 回归测试:在 `sys.path` 含 scripts/ 的前提下断言
   `importlib.import_module("train.train_drafter")` 解析到包路径
   (`train/__init__.py` 所在目录),并实调 `load_drafter_for_inference`
   的 import 段(CPU、仅 import 不加载权重)
3. 修复经 CPU 测试全绿后,冒烟重跑(同方案同预算,预计仍 <30 SU;
   本次 3.12 SU 计入损耗)

## 资源与状态

- 本次损耗:3.12 SU(总账:D2 线累计 G0a 10.13 + 冒烟失败 3.12)
- 显存:崩溃点 21.07GB(base+head 已载,drafter 未及加载)
- **T2(176390196)排队中,未受任何影响**;git 干净;
  lquota:scratch 4.77/10 TiB,inode ~647K 正常
- 等待授权:按上述方案修复 → 重跑冒烟
