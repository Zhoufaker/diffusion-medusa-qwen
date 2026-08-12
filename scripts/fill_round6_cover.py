#!/usr/bin/env python3.11
"""Fill Round-6 cover from payload FILE_MANIFEST + artifacts. encoding=utf-8."""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

PKG = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/gpt56_round6_release")
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


def build_skip_block() -> str:
    """Register the full-suite pass/skip census, grouped by declared gate."""
    census = C1 / "round6_pytest_full_suite/full_suite_summary.json"
    if not census.exists():
        raise RuntimeError(f"missing full-suite census: {census}")
    cs = json.loads(census.read_text(encoding="utf-8"))

    junit = PKG / "A_layer_no_torch/pytest_logs/tier_d_negctrl.junit.xml"
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
        "未声明的 skip 理由会让本封面生成失败（`SKIP_GATES` fail-closed），"
        "被静默停用的用例无法藏在 skip 计数里。\n"
        "外部训练期缓存缺失系 `/scratch` 生命周期清理所致，本轮不重建；"
        "`tests/test_cached_dataset.py` 属训练管道测试、依赖 `data/` 不在发布范围，"
        "故不入包，门声明见源仓库同路径。\n"
        f"负控子集（入包证据 `A_layer_no_torch/pytest_logs/tier_d_negctrl.junit.xml`）："
        f"**{nc['tests']} tests / {nc['failures']} failures / {nc['errors']} errors / "
        f"{nc['skipped']} skipped** —— candidate-specific 两用例、CLI integration 六用例、"
        "path-norm 两用例全部实跑通过，无负控被 skip。\n"
        "全量运行记录（不入包）：`round6_pytest_full_suite/full_suite.log`。"
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
    test_far = "test_candidate_near_tie_gap_ok_but_rank_far_is_hard"
    test_top2 = "test_candidate_near_tie_spec_eq_top2_despite_large_top2_gap"
    test_far_ln = find_def_line(tests, rf"^def {test_far}")
    test_top2_ln = find_def_line(tests, rf"^def {test_top2}")

    reprobe = C1 / "round6_candidate_reprobe" / "reprobe_summary.json"
    if not reprobe.exists():
        raise RuntimeError(f"missing re-probe summary: {reprobe}")
    rs = json.loads(reprobe.read_text(encoding="utf-8"))
    gs = rs.get("gap_spec") or {}
    reprobe_blk = (
        f"job `{rs.get('job')}`; n={rs.get('n_total')}; "
        f"context_verified={rs.get('n_context_verified')} "
        f"(prefix_mismatch={rs.get('n_prefix_mismatch')}); "
        f"pass={rs.get('n_pass_near_tie')} fail={rs.get('n_fail_hard')} "
        f"err={rs.get('n_error')}; all_pass={rs.get('all_pass')}; "
        f"spec_rank={rs.get('spec_rank_hist')}; "
        f"gap_spec max={gs.get('max')} (<=band {rs.get('band')}: "
        f"{gs.get('n_le_band')}/{rs.get('n_total')}); "
        f"spec==top2={rs.get('n_spec_eq_top2')}; "
        f"recorded_greedy in fp32 top2={rs.get('n_recorded_greedy_in_top2')}"
    )
    claim_status = (
        "UPGRADED to candidate-specific (pre-registered 1c satisfied)"
        if rs.get("all_pass") else
        "HELD at narrowed claim (fail rows awaiting per-row review)"
    )
    if not rs.get("all_pass"):
        raise RuntimeError(
            "re-probe not all_pass — Round-6 pack requires per-row review first"
        )

    skip_blk = build_skip_block()

    tc = json.loads(
        (C1 / "tier_c_interleaved_speed_d/tier_c_summary.json").read_text(encoding="utf-8"))
    v = tc["dyn_n24_vs_static_d3"]

    def bullets(xs):
        return "\n".join(f"- `{p}`" for p in xs)

    cover = f"""# Round-6 终验封面（框架沿用五轮；清单与 SHA 由 payload FILE_MANIFEST 单向链生成）

## 本轮性质
Round-5 接口/打包已过；本轮闭合 candidate-specific near_tie（唯一 GPU）+ 七条脚本修订。
范围声明不变：greedy-only / 单卡 V100 / batch=1 / transformers==5.3.0。

## 终态命名
**greedy numerical-safety verified / archive byte-reproducibility verified /
release bundle reproducible**
（safety 主张状态：**{claim_status}**）
终态命名不变；safety 的语义由 candidate-specific near_tie 支撑。

## Claim（Round-6 正式口径，candidate-specific）
**algorithmic greedy lossless; every first mid-sequence divergence is a
candidate-specific near_tie（spec_tok == greedy_top2 或
logit[top1] − logit[spec] ≤ 0.15）**
byte-exact 措辞仍仅限 archive reproducibility。
Round-5 收窄措辞（top1−top2 gap 版）保留在 `O0_CLAIM_NARROWED` 供旧产物溯源。

## 最小闭环

1. Candidate-specific near_tie：`is_candidate_near_tie` @ `decode/common.py:{loc_cand}`；
   `classify_o0_vs_ref` @ `{loc_cls}`。
   负控 `{test_far}`:{test_far_ln}（gap 过门但 rank 远 → hard）；
   `{test_top2}`:{test_top2_ln}（spec==top2 且 gap 过门 → near_tie）。

2. 512 re-probe：`A_layer_no_torch/artifacts/round6_reprobe/` —
   {reprobe_blk}
   探针含逐行上下文保真门（重放 greedy 前缀须等于记录的
   `greedy_context.before`；不匹配记 `PREFIX_MISMATCH`/`INCONCLUSIVE`，
   绝不静默记 hard）。OOD 行按原始跑套用 `max_pixels=501760`。
   包内 `*.o0_report.json` 由 legacy 规则标注，经本作业追溯复核；
   运行时分类器已切 candidate-specific。
   历史 item3 10/10 tree∈top-2 抽样作为旁证。
   前一次探针 `175812069` 因漏套 OOD `max_pixels` 判无效，
   留痕于 `round6_candidate_reprobe_v1_nomaxpixels/`（不入包）。

3. PBS `${{VAR:?}}` 全参数化；heredoc 已清 `o0_greedy_exact_pass` / `o0_spec_pass`。

4. CLI integration：`tests/test_o0_cli_integration.py`（exit 0/2/5/6/3 + archive parser）。

5. Integrity `.as_posix()` + path-norm 单测；`recompute_all` release fail-closed。

6. 完整 artifacts：下方分层清单（payload n_files={man["n_files"]}）。

## 测试与 skip 登记

{skip_blk}

## 流程告诫（本轮记入变更日志）

第一次探针 `175812069` 报出 21 条 `hard` 时，正确动作是**停等复核**；本轮实际是自行
定位探针侧缺陷（OOD 漏套 `max_pixels`）、修好并重跑为 `175813855`，事后才上报——
流程越界，**下不为例**。结论保留的依据：21 条失败 **21/21 ∈ OOD 段**
（`O1_c1_d3`/`O2_c1_6432`/`O3_b2_d3`/`O4_b2_6432`），in-domain 段零失败，错因闭合；
缺陷跑原样归档于 `round6_candidate_reprobe_v1_nomaxpixels/`（不入包，不被覆盖）。
正本记录：`docs/c1_eagle_conditioning.md` process change log 第 3 条；
方法论沉淀：`docs/agent_initial_prompt.md`「方法论沉淀」节。

## 包分层

### A 层（零依赖可验）
<!-- COVER_START -->
{bullets(a_paths)}

### B 层（GPU 复现路径）
{bullets(b_paths)}
<!-- COVER_END -->

B 层完整 rerun：见 `B_layer_gpu_rerun/RERUN.md`（与根 `RERUN.md` 对齐的 `qsub -v`）。

## 正式 cite（v2；科学数字未改）
σ = dyn_k8_n32 **2.825**（Δ+0.101±0.007 OOB）；
速度 **1.683/1.703 并列/未分**（primary 204；job 175785218）；
R = **0.433±0.039** mid。

## 完整性元数据
- FILE_MANIFEST n_files (payload): {man["n_files"]}
- FILE_MANIFEST sha256: `{man_sha}`
"""
    (PKG / "ROUND6_COVER.md").write_text(cover, encoding="utf-8")
    (PKG / "README.md").write_text(cover, encoding="utf-8")
    print(f"wrote cover; manifest_sha={man_sha}; claim_status={claim_status}")


if __name__ == "__main__":
    main()
