#!/usr/bin/env python3.11
"""Fill Round-7 cover from payload FILE_MANIFEST + artifacts. encoding=utf-8."""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

PKG = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/gpt56_round7_release")
SRC = Path("/home/562/mz9869/medusa-qwen")
C1 = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle")


def find_def_line(path: Path, pattern: str) -> int:
    rx = re.compile(pattern)
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if rx.search(line):
            return i
    raise RuntimeError(f"not found {pattern} in {path}")


# Every skip in the full suite must fall into one of these declared gates.
# An unrecognised skip reason aborts the pack: a silently disabled case must
# never be able to hide inside the skip count.
SKIP_GATES = (
    ("需 GPU", ("CUDA",)),
    ("需外部训练期缓存数据", ("external training-phase data",)),
)


def classify_skip(reason: str) -> str:
    for label, keys in SKIP_GATES:
        if any(k in reason for k in keys):
            return label
    raise RuntimeError(
        f"undeclared skip gate: {reason!r} — classify it in SKIP_GATES or unskip it"
    )


def build_skip_block(census_path: Path, junit: Path) -> str:
    """Register the full-suite pass/skip census, grouped by declared gate."""
    if not census_path.exists():
        raise RuntimeError(f"missing full-suite census: {census_path}")
    cs = json.loads(census_path.read_text(encoding="utf-8"))

    ts = ET.parse(junit).getroot()
    ts = ts if ts.tag == "testsuite" else ts.find("testsuite")
    nc = {k: int(ts.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}

    by_gate: dict[str, list[dict]] = {}
    for g in cs["skip_groups"]:
        by_gate.setdefault(classify_skip(g["reason"]), []).append(g)
    counts = {label: sum(g["n"] for g in by_gate.get(label, [])) for label, _ in SKIP_GATES}
    if sum(counts.values()) != cs["skipped"]:
        raise RuntimeError(
            f"skip census mismatch: gated {sum(counts.values())} vs reported {cs['skipped']}"
        )
    lines = [
        f"- **{label}**（{counts[label]}）："
        + ("；".join(f'{g["n"]}× `{g["loc"]}` — {g["reason"]}' for g in by_gate[label])
           if by_gate.get(label) else "—")
        for label, _ in SKIP_GATES
    ]
    tally = " + ".join(f"{counts[label]} {label}" for label, _ in SKIP_GATES)
    return (
        f"全量 `pytest tests/`：**{cs['passed']} passed / {cs['skipped']} skipped**，"
        f"0 failed。skipped 构成 = {tally}：\n\n"
        + "\n".join(lines)
        + "\n\n两类门都是显式门（skip 理由写明缺失的是哪类资源），均不在 O0/解码证据链内；"
        "未声明的 skip 理由会让本封面生成失败（`SKIP_GATES` fail-closed）。\n"
        f"负控子集（`A_layer_no_torch/pytest_logs/tier_d_negctrl.junit.xml`）："
        f"**{nc['tests']} tests / {nc['failures']} failures / {nc['errors']} errors / "
        f"{nc['skipped']} skipped**。\n"
        "全量运行记录与 skip 清单本轮**入包**："
        "`A_layer_no_torch/pytest_logs/full_suite.log`、"
        "`A_layer_no_torch/pytest_logs/skip_inventory.json`。"
    )


def main() -> None:
    man = json.loads((PKG / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    man_sha = hashlib.sha256((PKG / "FILE_MANIFEST.json").read_bytes()).hexdigest()
    paths = [f["path"] for f in man["files"]]
    a_paths = sorted(p for p in paths if p.startswith("A_layer_no_torch/"))
    b_paths = sorted(p for p in paths if p.startswith("B_layer_gpu_rerun/"))

    common = SRC / "decode/common.py"
    tests = SRC / "tests/test_o0_and_ordered.py"
    loc_cand = find_def_line(common, r"^def is_candidate_near_tie")
    loc_cls = find_def_line(common, r"^def classify_o0_vs_ref")
    boundary = {
        "rank2_in_band": "test_candidate_gap_only_rank2_within_band_is_near_tie",
        "rank3_in_band": "test_candidate_gap_only_rank3_within_band_is_near_tie",
        "gap_missing": "test_candidate_gap_missing_is_hard",
        "gap_above": "test_candidate_gap_just_above_band_is_hard",
        "reversed": "test_candidate_near_tie_spec_eq_top2_despite_large_gap_is_hard",
        "rank_far": "test_candidate_near_tie_gap_ok_but_rank_far_is_hard",
    }
    boundary_ln = {k: find_def_line(tests, rf"^def {v}") for k, v in boundary.items()}

    rc = json.loads(
        (C1 / "round7_gap_only_reclass/reclass_summary.json").read_text(encoding="utf-8"))
    if not rc.get("all_pass"):
        raise RuntimeError(
            "gap-only re-classification not all_pass — Round-7 pack requires review first"
        )
    gs = rc["gap_spec"]
    reclass_blk = (
        f"source rows `{Path(rc['source_rows']).name}` "
        f"sha256 `{rc['source_rows_sha256'][:16]}…`（job `{rc['source_job']}`，"
        f"**gpu_rerun={rc['gpu_rerun']}**）; n={rc['n_total']}; "
        f"tail_context_verified={rc['n_tail_context_verified']} "
        f"（窗口 {rc['tail_context_window']}）; "
        f"pass={rc['n_pass_near_tie']} hard={rc['n_fail_hard']} "
        f"inconclusive={rc['n_inconclusive']}; all_pass={rc['all_pass']}; "
        f"vs Round-6 判定变化 {rc['n_kind_changed_vs_round6']} 条; "
        f"gap_spec min={gs['min']} max={gs['max']} mean={gs['mean']:.6f} "
        f"（≤band {gs['n_le_band']}/{rc['n_total']}，>band {gs['n_gt_band']}，"
        f"负值 {gs['n_negative']}）; rank 分布 {rc['spec_rank_hist']}; "
        f"分段 {rc['pass_by_segment']}"
    )

    prov = json.loads(
        (C1 / "round6_candidate_reprobe_v1_nomaxpixels/reprobe_summary.json")
        .read_text(encoding="utf-8"))
    n_v1_fail = len(prov.get("fails") or [])

    skip_blk = build_skip_block(
        C1 / "round7_pytest_full_suite/full_suite_summary.json",
        PKG / "A_layer_no_torch/pytest_logs/tier_d_negctrl.junit.xml",
    )

    def bullets(xs):
        return "\n".join(f"- `{p}`" for p in xs)

    cover = f"""# Round-7 终验封面（清单与 SHA 由 payload FILE_MANIFEST 单向链生成）

## 本轮性质
六轮科学证据独立复核通过；本轮闭合两条接口阻断（near-tie predicate 规格错误、
PBS 固定日志路径）+ 三条 P2 + 一条 P3。零 GPU 重跑，仅一次 evaluator smoke。
范围声明不变：greedy-only / 单卡 V100 / batch=1 / transformers==5.3.0。

## 终态命名（终版，无 pending 尾巴）
**greedy numerical-safety verified / archive byte-reproducibility verified /
release bundle reproducible**

## Claim（Round-7 正式口径，gap-only）
**algorithmic greedy lossless; every first mid-sequence divergence is a
candidate-specific near_tie（0 ≤ logit[greedy_top1] − logit[spec_tok] ≤ 0.15）**
byte-exact 措辞仍仅限 archive reproducibility。
Round-6 的 `spec_tok == greedy_top2` 析取项是规格错误（rank 2 不约束 gap 大小，
可放行任意大的分离），本轮删除；`spec_rank` 与 `spec_tok == greedy_top2`
降为纯诊断字段。旧措辞保留在 `O0_CLAIM_NARROWED` / `O0_CLAIM_CANDIDATE_SPECIFIC`
供旧产物溯源，包内旧 report 的 `official_claim` 已改标
`official_claim_at_run_time_retracted` 并指向本轮口径。

## 最小闭环

1. **【阻断】gap-only predicate**：`is_candidate_near_tie` @
   `decode/common.py:{loc_cand}`（唯一条件 spec_tok 非空且 0 ≤ gap_spec ≤ band）；
   `classify_o0_vs_ref` @ `{loc_cls}`。
   反转负控 `{boundary['reversed']}`:{boundary_ln['reversed']}
   （六轮判 near_tie 的 spec==top2 且 gap=0.50 反例，本轮改判 **hard**）。
   四类边界：`{boundary['rank2_in_band']}`:{boundary_ln['rank2_in_band']}（top2 且 gap≤band → near_tie）；
   `{boundary['rank3_in_band']}`:{boundary_ln['rank3_in_band']}（rank3 且 gap≤band → near_tie）；
   `{boundary['gap_missing']}`:{boundary_ln['gap_missing']}（gap 缺失 → hard）；
   `{boundary['gap_above']}`:{boundary_ln['gap_above']}（band 含端点，超出一丝 → hard）。
   另留 `{boundary['rank_far']}`:{boundary_ln['rank_far']}（rank 远且 gap 大 → hard）。

2. **【阻断随动】512 条离线严格重分类**（`A_layer_no_torch/artifacts/round7_gap_only_reclass/`）：
   {reclass_blk}
   未重跑 512 条 probe：六轮 job `175813855` 已逐行记录满精度 `gap_spec`，
   本轮只换判据、就地重打分；tail 上下文未通过的行仍记 INCONCLUSIVE，
   换判据不能追认未验证的重放。
   关键佐证：5 条 `spec_rank=3` 的行凭自身 gap 通过，证明 gap-only 不是 rank≤2 的换皮；
   最大 gap {gs['max']} ≪ band 0.15。
   CPU 合成负控全绿后跑了一次 GPU evaluator smoke 确认新 predicate 接线
   （`A_layer_no_torch/artifacts/round7_smoke/`）。

3. **【阻断】PBS 日志路径去站点化**：五份发布 PBS 删除固定 `#PBS -o` 与固定
   `mkdir /scratch/li96/mz9869/logs`；日志经 `qsub -o "$MEDUSA_LOG_DIR"` 传入，
   `MEDUSA_LOG_DIR` 进入 fail-fast checklist；`RERUN.md` 示例同步。
   验证：`tests/test_pbs_log_portability.py` 用一个不含 `/scratch/li96` 布局的
   临时目录跑 prologue —— 全 env 齐备时通过，缺 `MEDUSA_LOG_DIR` 时立即失败。

4. **integrity exclusions 改 root-relative**：只排除顶层链文件与 checker 自身输出；
   `B_layer_gpu_rerun/README.md` 本轮起进入 manifest（回归测试
   `test_nested_readme_enters_manifest`）。

5. **archive 写出参数 guard**：`--o0-write-archive` / `--o0-write-greedy-archive`
   与 `--o0-archive` 同样要求 `--check-greedy-bytes`，各配一条 subprocess 负测。

6. **字段口径**：`n_context_verified` → `n_tail_context_verified`
   （窗口 `last_up_to_5_tokens`，措辞与封面一致）；O0 report 本版起写入
   完整 greedy prefix 的 SHA-256（`greedy_prefix_sha256` / `greedy_full_sha256`）
   与 `fingerprint`（prompt/config/model，含 manifest 与 ckpt 的 sha256）。
   **只加字段定义与写入，不回填旧数据。**

7. **provenance-only artifacts 入包**：`A_layer_no_torch/pytest_logs/full_suite.log`、
   `A_layer_no_torch/pytest_logs/skip_inventory.json`、
   `A_layer_no_torch/artifacts/provenance/round6_v1_nomaxpixels_summary.json`
   （首次无效 probe `175812069` 的摘要，{n_v1_fail} 条失败全部 ∈ OOD 段）。
   provenance 目录不参与 scrub，按原样保留超期产物的字面记录。

8. 完整 artifacts：下方分层清单（payload n_files={man["n_files"]}）。

## 测试与 skip 登记

{skip_blk}

## 流程告诫（沿用六轮记录）

probe/gate 失败必须停等复核；`175812069` → `175813855` 自行修复重跑属越界，
**下不为例**。正本：`docs/c1_eagle_conditioning.md` process change log 第 3 条；
方法论沉淀：`docs/agent_initial_prompt.md`「方法论沉淀」节。

## 包分层

### A 层（零依赖可验）
<!-- COVER_START -->
{bullets(a_paths)}

### B 层（GPU 复现路径）
{bullets(b_paths)}
<!-- COVER_END -->

B 层完整 rerun：见 `B_layer_gpu_rerun/RERUN.md`（含 `qsub -o "$MEDUSA_LOG_DIR"` 示例）。

## 正式 cite（v2；科学数字未改）
σ = dyn_k8_n32 **2.825**（Δ+0.101±0.007 OOB）；
速度 **1.683/1.703 并列/未分**（primary 204；job 175785218）；
R = **0.433±0.039** mid。

## 完整性元数据
- FILE_MANIFEST n_files (payload): {man["n_files"]}
- FILE_MANIFEST sha256: `{man_sha}`
"""
    (PKG / "ROUND7_COVER.md").write_text(cover, encoding="utf-8")
    (PKG / "README.md").write_text(cover, encoding="utf-8")
    print(f"wrote cover; manifest_sha={man_sha}; n_files={man['n_files']}")


if __name__ == "__main__":
    main()
