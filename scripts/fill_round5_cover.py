#!/usr/bin/env python3.11
"""Fill Round-5 cover from payload FILE_MANIFEST + artifacts. encoding=utf-8."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PKG = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/gpt56_round5_release")
SRC = Path("/home/562/mz9869/medusa-qwen")
C1 = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle")


def find_def_line(path: Path, pattern: str) -> int:
    rx = re.compile(pattern)
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if rx.search(line):
            return i
    raise RuntimeError(f"not found {pattern} in {path}")


def main() -> None:
    man = json.loads((PKG / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    man_sha = hashlib.sha256((PKG / "FILE_MANIFEST.json").read_bytes()).hexdigest()
    paths = [f["path"] for f in man["files"]]
    a_paths = sorted(p for p in paths if p.startswith("A_layer_no_torch/"))
    b_paths = sorted(p for p in paths if p.startswith("B_layer_gpu_rerun/"))

    common = SRC / "decode/common.py"
    tests = SRC / "tests/test_o0_and_ordered.py"
    loc_byte = find_def_line(common, r"^def greedy_byte_exact_pass")
    loc_safe = find_def_line(common, r"^def greedy_numerical_safety_pass")
    loc_arch = find_def_line(common, r"^def archive_gate_status")
    loc_exit = find_def_line(common, r"^def archive_gate_exit_code")
    test_byte = "test_byte_exact_false_when_near_tie_present"
    test_zero = "test_archive_gate_zero_coverage_not_run"
    test_exit0 = "test_archive_explicit_zero_coverage_nonzero_exit"
    test_exit_p = "test_archive_explicit_partial_coverage_nonzero_exit"
    test_byte_ln = find_def_line(tests, rf"^def {test_byte}")
    test_zero_ln = find_def_line(tests, rf"^def {test_zero}")
    test_exit0_ln = find_def_line(tests, rf"^def {test_exit0}")
    test_exit_p_ln = find_def_line(tests, rf"^def {test_exit_p}")

    repro = json.loads(
        (C1 / "tier_d_repro_fullcover/repro_summary.json").read_text(encoding="utf-8"))
    exact_bits = [
        f"{tag}: n_exact={r['n_exact']}/{r['n_prompts']} near_tie={r['n_near_tie']}"
        for tag, r in list(repro["rows"].items())[:3]
    ]
    tc = json.loads(
        (C1 / "tier_c_interleaved_speed_d/tier_c_summary.json").read_text(encoding="utf-8"))
    v = tc["dyn_n24_vs_static_d3"]
    vs = (tc.get("sensitivity_non_identical") or {}).get("dyn_n24_vs_static_d3") or {}
    va = (tc.get("all_prompts") or {}).get("dyn_n24_vs_static_d3") or {}

    def bullets(xs):
        return "\n".join(f"- `{p}`" for p in xs)

    cover = f"""# Round-5 终验封面（框架沿用四轮；清单与 SHA 由 payload FILE_MANIFEST 单向链生成）

## 本轮性质
Round-4 科学证据已终审通过；本轮闭环接口/打包八条修订。
范围声明不变：greedy-only / 单卡 V100 / batch=1 / transformers==5.3.0。

## 终态命名
**greedy numerical-safety verified / archive byte-reproducibility verified /
release bundle reproducible**

## Round-3/4 最小闭环 → 闭合位置

1. O0 双 verdict：`decode/common.py:{loc_byte}` / `{loc_safe}`；
   负控 `{test_byte}` @ `tests/test_o0_and_ordered.py:{test_byte_ln}`。
   `greedy_byte_exact_pass` 如实 FALSE（例：{exact_bits[0]}；{exact_bits[1]}）。
   无 `greedy_exact_*` 机器字段；唯一 deprecated 别名
   `legacy_numerical_safety_pass_deprecated`。

2. archive gate 四态 + 显式 archive 非零退出：`archive_gate_status` @
   `decode/common.py:{loc_arch}`；`archive_gate_exit_code` @
   `decode/common.py:{loc_exit}`（NOT_RUN→5，INCOMPLETE→6，FAIL→2）；
   负控 `{test_zero}`:{test_zero_ln}；`{test_exit0}`:{test_exit0_ln}；
   `{test_exit_p}`:{test_exit_p_ln}。

3. reproducibility 全覆盖：job `{repro["job"]}`；
   汇总 `A_layer_no_torch/artifacts/tier_d_repro/repro_summary.json`
   （`{repro["overall"]}`，n={len(repro["rows"])}）。

4. Tier C：raw `A_layer_no_torch/artifacts/tier_c/tier_c_summary_raw.json`；
   hash 字段 `token_hash`；官方 cite = hash-rerun primary n={tc["n_byte_identical_primary"]}/300
   static={tc["static_d3_speedup_mean"]:.3f} / dyn={tc["dyn_n24_speedup_mean"]:.3f}；
   Δ={v["delta_mean"]:+.4f}±{v["delta_se"]:.4f}；verdict=`{v["verdict"]}`；
   sensitivity n={tc.get("n_sensitivity")} verdict=`{vs.get("verdict")}`；
   all-prompts verdict=`{va.get("verdict")}`（并列报）。

5. 完整 artifacts：下方分层清单（payload n_files={man["n_files"]}）。

6. 完整性链：`INTEGRITY_CHECK.json`（`check_release_integrity.py`）。

## 包分层

### A 层（零依赖可验）
<!-- COVER_START -->
{bullets(a_paths)}

### B 层（GPU 复现路径）
{bullets(b_paths)}
<!-- COVER_END -->

pytest：`A_layer_no_torch/pytest_logs/tier_d_negctrl.log` +
`A_layer_no_torch/pytest_logs/tier_d_negctrl.junit.xml` +
`A_layer_no_torch/pytest_logs/env_fingerprint.json`。

B 层完整 rerun（解压后）::

    export MEDUSA_PROJECT_ROOT="$PWD/B_layer_gpu_rerun"
    export HF_HOME=/path/to/hf_cache
    pip install -r B_layer_gpu_rerun/requirements.lock
    pip install -e B_layer_gpu_rerun   # 或: pip install ./B_layer_gpu_rerun
    cd "$MEDUSA_PROJECT_ROOT" && pytest tests/test_o0_and_ordered.py tests/test_max_new_cap.py -q
    # 全覆盖 repro（外部数据路径显式传入）:
    qsub -v MEDUSA_PROJECT_ROOT="$MEDUSA_PROJECT_ROOT" pbs/tier_d_repro_fullcover.pbs

## 正式 cite（v2）
σ = dyn_k8_n32 **2.825**（Δ+0.101±0.007 OOB）；
速度 **1.683/1.703 并列/未分**（primary 204；job 175785218）；
R = **0.433±0.039** mid；
实现级等价 = numerical_safety（byte_exact 如实 FALSE；byte-exact 仅 archive）。

## 完整性元数据
- FILE_MANIFEST n_files (payload): {man["n_files"]}
- FILE_MANIFEST sha256: `{man_sha}`
"""
    (PKG / "ROUND5_COVER.md").write_text(cover, encoding="utf-8")
    (PKG / "README.md").write_text(cover, encoding="utf-8")
    print(f"wrote cover; manifest_sha={man_sha}")


if __name__ == "__main__":
    main()
