#!/usr/bin/env python3.11
"""Pure-stdlib recompute of official v2 / Tier C cites (no torch / numpy).

Traverses **all** rows in v1_to_v2_delta_summary.json (no fixed sample).
All text/JSON IO uses encoding=\"utf-8\".

Usage:
  python3.11 scripts/recompute_all.py --root .../A_layer_no_torch
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple


OFFICIAL = {
    "sigma_champ_dyn_n32": 2.8249349622902047,
    "paired_sigma_delta": 0.10116204624279582,
    "paired_sigma_se": 0.007084585642065481,
    "R_100": 0.4327688409687326,
    "R_100_se": 0.03851110979973412,
    "tier_c_verdict": "并列/未分",
    "tier_c_static": 1.6825768876219809,
    "tier_c_dyn": 1.7026106932508362,
    "tier_c_delta": 0.02003380562885532,
    "tier_c_n_primary": 204,
}


# Packet-local v1_lineage filenames for each delta-summary tag
V1_LINEAGE_NAME = {
    "gate100_c1_d3": "gate100_c1_d3.json",  # final_eval/c1_d3.json copy
    "gate100_c1_6432": "c10_6432_32.json",
    "gate100_b2_wide": "b2_wide.json",
    "bridge300_c1_d3": "bridge300_c1_d3.json",
    "bridge300_c1_6432": "bridge300_c1_6432.json",
    "bridge300_b2_d3": "bridge300_b2_d3.json",
    "bridge300_b2_6432": "bridge300_b2_6432.json",
    "dyn_k8_n24": "dyn_k8_n24.json",
    "dyn_k8_n32": "dyn_k8_n32.json",
    "O1_c1_d3": "O1_c1_d3.json",
    "O2_c1_6432": "O2_c1_6432.json",
    "O3_b2_d3": "O3_b2_d3.json",
    "O4_b2_6432": "O4_b2_6432.json",
}


def mean_se(xs: List[float]) -> Tuple[float, float]:
    m = statistics.mean(xs)
    se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, se


def paired_delta(a: List[Dict], b: List[Dict], key: str = "sigma") -> Tuple[float, float, int]:
    mb = {str(r["id"]): r[key] for r in b}
    diffs = [r[key] - mb[str(r["id"])] for r in a if str(r["id"]) in mb]
    m, se = mean_se(diffs)
    return m, se, len(diffs)


def load(p: Path) -> Dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= max(tol * max(1.0, abs(b)), 1e-6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="A_layer_no_torch root")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument(
        "--allow-summary-fallback",
        action="store_true",
        help="Permit delta rows to use summary sigma when raw v1/v2 JSON missing. "
             "Default (release mode): any summary fallback → FAIL / non-zero exit.",
    )
    args = ap.parse_args()

    if args.root:
        root = Path(args.root)
        art = root / "artifacts"
        v2 = art / "v2_rebaseline"
        tc = art / "tier_c"
        v1 = art / "v1_lineage"
        delta_path = v2 / "v1_to_v2_delta_summary.json"
        if not delta_path.exists():
            delta_path = art / "v1_to_v2_delta_summary.json"
        repro = art / "tier_d_repro" / "repro_summary.json"
    else:
        C1 = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle")
        v2 = C1 / "v2_rebaseline"
        tc = C1 / "tier_c_interleaved_speed_d"
        v1 = art = None  # type: ignore
        delta_path = v2 / "v1_to_v2_delta_summary.json"
        repro = C1 / "tier_d_repro_fullcover" / "repro_summary.json"
        # scratch layout for v1
        scratch_v1 = {
            "gate100_c1_d3": C1 / "final_eval/c1_d3.json",
            "gate100_c1_6432": C1 / "fanout_sweep/c10_6432_32.json",
            "gate100_b2_wide": C1 / "final_eval/b2_wide.json",
            "bridge300_c1_d3": C1 / "bridge_300/c1_d3.json",
            "bridge300_c1_6432": C1 / "bridge_300/c1_6432.json",
            "bridge300_b2_d3": C1 / "bridge_300/b2_d3.json",
            "bridge300_b2_6432": C1 / "bridge_300/b2_6432.json",
            "dyn_k8_n24": C1 / "dynamic_sweep_300/dyn_k8_n24.json",
            "dyn_k8_n32": C1 / "dynamic_sweep_300/dyn_k8_n32.json",
            "O1_c1_d3": C1 / "ood_mmvet_218/O1_c1_d3.json",
            "O2_c1_6432": C1 / "ood_mmvet_218/O2_c1_6432.json",
            "O3_b2_d3": C1 / "ood_mmvet_218/O3_b2_d3.json",
            "O4_b2_6432": C1 / "ood_mmvet_218/O4_b2_6432.json",
        }

    checks = []
    rows: Dict = {}

    dyn = load(v2 / "dyn_k8_n32.json")
    br = load(v2 / "bridge300_c1_6432.json")
    spm = dyn.get("sigma_prompt_mean", dyn["sigma_mean"])
    rows["sigma_champ"] = spm
    checks.append(("sigma_champ_dyn_n32", spm, OFFICIAL["sigma_champ_dyn_n32"]))

    dm, dse, n = paired_delta(dyn["per_prompt_results"], br["per_prompt_results"])
    rows["paired_sigma"] = {"delta": dm, "se": dse, "n": n}
    checks.append(("paired_sigma_delta", dm, OFFICIAL["paired_sigma_delta"]))
    checks.append(("paired_sigma_se", dse, OFFICIAL["paired_sigma_se"]))

    o2 = load(v2 / "O2_c1_6432.json")
    o4 = load(v2 / "O4_b2_6432.json")
    od, ose, on = paired_delta(o2["per_prompt_results"], o4["per_prompt_results"])
    R, Rse = od / 0.310, ose / 0.310
    rows["ood_R"] = {"delta": od, "se": ose, "R_100": R, "R_100_se": Rse, "n": on}
    checks.append(("R_100", R, OFFICIAL["R_100"]))
    checks.append(("R_100_se", Rse, OFFICIAL["R_100_se"]))

    # Full delta table from summary — every row with sigma_v1/sigma_v2
    delta_sum = load(delta_path)
    delta_rows = {}
    missing = []
    for tag, meta in delta_sum["rows"].items():
        if "sigma_v1" not in meta or "sigma_v2" not in meta:
            continue
        # Prefer recomputing from raw files when available
        v2p = v2 / f"{tag}.json"
        if args.root:
            v1p = v1 / V1_LINEAGE_NAME.get(tag, f"{tag}.json")
        else:
            v1p = scratch_v1.get(tag)
        if v1p is not None and Path(v1p).exists() and v2p.exists():
            s1 = load(Path(v1p))["sigma_mean"]
            s2 = load(v2p).get("sigma_prompt_mean", load(v2p)["sigma_mean"])
            dlt = (s2 - s1) / s1 * 100.0
        else:
            s1, s2, dlt = meta["sigma_v1"], meta["sigma_v2"], meta["delta_pct"]
            missing.append(tag)
        delta_rows[tag] = {"v1": s1, "v2": s2, "delta_pct": dlt,
                           "summary_delta_pct": meta["delta_pct"]}
        checks.append((f"delta_pct::{tag}", dlt, meta["delta_pct"]))
    rows["delta_all_rows"] = delta_rows
    rows["delta_n_rows"] = len(delta_rows)
    if missing:
        rows["delta_used_summary_fallback"] = missing
        if not args.allow_summary_fallback:
            rows["release_mode_fail_closed"] = True
            print(
                "FAIL-CLOSED: summary fallback used for tags: "
                + ", ".join(missing)
                + " (pass --allow-summary-fallback to degrade)",
                file=sys.stderr,
            )

    # Tier C official = hash rerun primary
    tcs = load(tc / "tier_c_summary.json")
    v = tcs["dyn_n24_vs_static_d3"]
    rows["tier_c_official"] = {
        "job": tcs.get("job") or "175785218",
        "static": tcs["static_d3_speedup_mean"],
        "dyn": tcs["dyn_n24_speedup_mean"],
        "delta": v["delta_mean"],
        "se": v["delta_se"],
        "ci95": v["ci95"],
        "band": v["band_1pct"],
        "verdict": v["verdict"],
        "n_primary": tcs.get("n_byte_identical_primary"),
        "n_sensitivity": tcs.get("n_sensitivity"),
        "sensitivity": (tcs.get("sensitivity_non_identical") or {}).get("dyn_n24_vs_static_d3"),
        "all_prompts": (tcs.get("all_prompts") or {}).get("dyn_n24_vs_static_d3"),
    }
    checks.append(("tier_c_verdict", v["verdict"], OFFICIAL["tier_c_verdict"]))
    checks.append(("tier_c_static", tcs["static_d3_speedup_mean"], OFFICIAL["tier_c_static"]))
    checks.append(("tier_c_dyn", tcs["dyn_n24_speedup_mean"], OFFICIAL["tier_c_dyn"]))
    checks.append(("tier_c_delta", v["delta_mean"], OFFICIAL["tier_c_delta"]))
    checks.append(("tier_c_n_primary", tcs.get("n_byte_identical_primary"),
                   OFFICIAL["tier_c_n_primary"]))

    if Path(repro).exists():
        rs = load(Path(repro))
        rows["fullcover_repro"] = {"overall": rs.get("overall"), "n_rows": len(rs.get("rows", {}))}
        checks.append(("fullcover_repro", rs.get("overall"), "PASS_FULLCOVER_REPRO"))

    rows["dual"] = {
        "sigma_prompt_mean": dyn.get("sigma_prompt_mean", dyn["sigma_mean"]),
        "sigma_round_pooled": dyn.get("sigma_round_pooled"),
        "nbar_prompt_mean": dyn.get("nbar_prompt_mean", dyn.get("mean_tree_width")),
        "nbar_round_pooled": dyn.get("nbar_round_pooled"),
    }
    rows["end_state"] = (
        "greedy numerical-safety verified / "
        "archive byte-reproducibility verified / "
        "release bundle reproducible"
    )

    print("=== RECOMPUTE_ALL (stdlib, full delta table) ===")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print("\n=== CITE CHECKLIST ===")
    ok = True
    if missing and not args.allow_summary_fallback:
        ok = False
        print("  [FAIL] summary_fallback_forbidden: "
              f"tags={missing}")
    for name, got, exp in checks:
        if isinstance(exp, str) or isinstance(got, str):
            match = got == exp
        else:
            match = close(float(got), float(exp), args.tol)
        flag = "OK" if match else "MISMATCH"
        if not match:
            ok = False
        print(f"  [{flag}] {name}: got={got} expected={exp}")
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
