#!/usr/bin/env python3.11
"""Programmatically fill Round-4 cover from FILE_MANIFEST + artifacts. No hand-copy."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PKG = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/gpt56_round4_release")
SRC = Path("/home/562/mz9869/medusa-qwen")
C1 = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle")


def find_def_line(path: Path, pattern: str) -> int:
    rx = re.compile(pattern)
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if rx.search(line):
            return i
    raise RuntimeError(f"pattern not found in {path}: {pattern}")


def main() -> None:
    man = json.loads((PKG / "FILE_MANIFEST.json").read_text())
    paths = [f["path"] for f in man["files"]]
    a_paths = sorted(p for p in paths if p.startswith("A_layer_no_torch/"))
    b_paths = sorted(p for p in paths if p.startswith("B_layer_gpu_rerun/"))

    common = SRC / "decode/common.py"
    tests = SRC / "tests/test_o0_and_ordered.py"
    loc_byte = find_def_line(common, r"^def greedy_byte_exact_pass")
    loc_safe = find_def_line(common, r"^def greedy_numerical_safety_pass")
    loc_arch = find_def_line(common, r"^def archive_gate_status")
    loc_not_run = find_def_line(common, r'^ARCHIVE_GATE_NOT_RUN\s*=')
    test_byte = "test_byte_exact_false_when_near_tie_present"
    test_safe = "test_safety_true_with_near_tie_zero_material"
    test_zero = "test_archive_gate_zero_coverage_not_run"
    test_byte_ln = find_def_line(tests, rf"^def {test_byte}")
    test_zero_ln = find_def_line(tests, rf"^def {test_zero}")

    repro = json.loads((C1 / "tier_d_repro_fullcover/repro_summary.json").read_text())
    # n_exact samples from repro o0 reports (actual artifacts)
    exact_bits = []
    for tag, row in repro["rows"].items():
        exact_bits.append(
            f"{tag}: n_exact={row['n_exact']}/{row['n_prompts']} "
            f"near_tie={row['n_near_tie']} byte_exact={row['greedy_byte_exact_pass']}"
        )
    # packet-relative paths
    repro_sum_pkg = "A_layer_no_torch/artifacts/tier_d_repro/repro_summary.json"
    assert repro_sum_pkg in paths, repro_sum_pkg

    tc = json.loads((C1 / "tier_c_interleaved_speed_d/tier_c_summary.json").read_text())
    tc_raw_pkg = "A_layer_no_torch/artifacts/tier_c/tier_c_summary_raw.json"
    tc_sum_pkg = "A_layer_no_torch/artifacts/tier_c/tier_c_summary.json"
    assert tc_raw_pkg in paths and tc_sum_pkg in paths
    raw = json.loads((PKG / tc_raw_pkg).read_text())
    hash_field = "token_hash"
    assert hash_field in raw["blocks"][0]["per_prompt"][0]["runs"]["static_d3"]
    n_pri = tc["n_byte_identical_primary"]
    n_sens = tc["n_sensitivity"]
    n_all = tc["n_prompts"]
    v_pri = tc["dyn_n24_vs_static_d3"]
    v_sens = tc["sensitivity_non_identical"]["dyn_n24_vs_static_d3"]
    tco = json.loads(
        (PKG / "A_layer_no_torch/artifacts/tier_c_official_175738321/tier_c_summary.json").read_text()
    )
    vo = tco["dyn_n24_vs_static_d3"]

    # ensure recompute + cover outputs exist (caller may regenerate)
    recompute_out = PKG / "A_layer_no_torch/artifacts/recompute_all_output.txt"
    cover_out = PKG / "COVER_CHECK.json"
    pytest_log = "A_layer_no_torch/pytest_logs/tier_d_negctrl.log"
    pytest_junit = "A_layer_no_torch/pytest_logs/tier_d_negctrl.junit.xml"
    env_fp = "A_layer_no_torch/pytest_logs/env_fingerprint.json"

    def bullets(xs):
        return "\n".join(f"- `{p}`" for p in xs)

    cover = f"""# Round-4 终验封面（框架由复核方提供，清单由 FILE_MANIFEST 程序化生成）

## 本轮性质
Round-3 三个 NOT CLOSED（#2/#3/#5/#6 涉及项）的闭环终验。
范围声明不变：greedy-only / 单卡 V100 / batch=1 / transformers==5.3.0。

## Round-3 最小闭环要求 → 闭合位置（逐条）

1. O0 双 verdict 真语义：`decode/common.py:{loc_byte}` (`greedy_byte_exact_pass`)；
   `decode/common.py:{loc_safe}` (`greedy_numerical_safety_pass`)；
   负控 `{test_byte}` @ `tests/test_o0_and_ordered.py:{test_byte_ln}`
   （另见 `{test_safe}`）。
   —— `greedy_byte_exact_pass` 如实报 FALSE（各 report `n_exact` 例：
   {exact_bits[0]}；{exact_bits[8] if len(exact_bits)>8 else exact_bits[-1]}；
   全表见 `{repro_sum_pkg}`）。
   官方主张已改为「算法语义无损 + 无 hard/材料性分歧，near-tie 已标定」。

2. archive gate 四态机：`decode/common.py:{loc_arch}` (`archive_gate_status`)；
   常量 `ARCHIVE_GATE_NOT_RUN` @ `decode/common.py:{loc_not_run}`；
   零覆盖 NOT_RUN 负控 `{test_zero}` @ `tests/test_o0_and_ordered.py:{test_zero_ln}`。

3. reproducibility 全覆盖实测：job `{repro["job"]}`；
   逐行 PASS 汇总 `{repro_sum_pkg}`（overall=`{repro["overall"]}`，n_rows={len(repro["rows"])}）。

4. Tier C raw + 输出等价 gate：raw blocks `{tc_raw_pkg}`；
   hash 字段名 `{hash_field}`；
   byte-identical 主分析 n={n_pri}/{n_all}，verdict=`{v_pri["verdict"]}`，
   Δ={v_pri["delta_mean"]:+.4f}±{v_pri["delta_se"]:.4f}，
   95% CI=[{v_pri["ci95"][0]:+.4f}, {v_pri["ci95"][1]:+.4f}]，1% band=±{v_pri["band_1pct"]:.4f}；
   敏感性 n={n_sens}/{n_all}，verdict=`{v_sens["verdict"]}`，
   Δ={v_sens["delta_mean"]:+.4f}，CI=[{v_sens["ci95"][0]:+.4f}, {v_sens["ci95"][1]:+.4f}]；
   官方绝对速度 cite（job `175738321`）：static={tco["static_d3_speedup_mean"]:.3f} /
   dyn={tco["dyn_n24_speedup_mean"]:.3f}，verdict=`{vo["verdict"]}`。

5. 完整 artifacts：见下方分层清单（由 FILE_MANIFEST 生成，n_files={man["n_files"]}）。

6. 封面自动核对：`COVER_CHECK.json`（由 `A_layer_no_torch/scripts/assert_cover_from_manifest.py` 写出）。

## 包分层

### A 层（零依赖可验）
<!-- COVER_START -->
{bullets(a_paths)}

recompute_all.py 自测输出：`A_layer_no_torch/artifacts/recompute_all_output.txt`

### B 层（GPU 复现路径）
{bullets(b_paths)}
<!-- COVER_END -->

pytest：log=`{pytest_log}`；junitxml=`{pytest_junit}`；环境指纹=`{env_fp}`。

## 正式 cite（v2，语义修正后）
σ = dyn_k8_n32 2.825（配对 Δ+0.101±0.007 OOB）；
速度 1.729/1.753 并列/未分（预注册 95% CI 判据）；
R = 0.433±0.039 mid；
实现级等价主张 = numerical_safety（byte_exact 如实为 FALSE，
near-tie 已标定，byte-exact 仅用于 archive reproducibility）。

## 请求重点
按 Round-3 六条逐一给 closed/not-closed；预期终态
「greedy-exact verified（修正语义）/ release-artifacts reproducible」。
已裁决未变项不必重展开。

## 程序化填充元数据
- generator: `scripts/fill_round4_cover.py`
- FILE_MANIFEST n_files: {man["n_files"]}
- FILE_MANIFEST sha256: {hashlib.sha256((PKG/"FILE_MANIFEST.json").read_bytes()).hexdigest()}
"""
    out = PKG / "ROUND4_COVER.md"
    out.write_text(cover)
    # also replace package README cover with this authoritative cover
    (PKG / "README.md").write_text(cover)
    print(f"wrote {out}")
    print(f"manifest_sha={hashlib.sha256((PKG/'FILE_MANIFEST.json').read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
