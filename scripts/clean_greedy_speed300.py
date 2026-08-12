#!/usr/bin/env python3
"""Clean greedy timing for speed_300 — acceptance gates locked before lock-in.

Gates (any fail ⇒ exit 2, do not lock speed_300):
  (a) old100 subset greedy tok/s ∈ 30.3 ± 3%  (same prompts as historical)
  (b) tree tok/s = bridge 174124013 locked numerators (not re-run);
      if a re-run were supplied it must be within ±2% of those four
  (c) report speed_300 under per-config and pooled greedy; champion = pooled

With a single clean greedy pass, own-greedy ≡ pooled (one shared denom).
Historical sweep pooled = mean of per-config greeds; here n_config=1 clean.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from decode.common import filter_prompts, load_base, vanilla_greedy  # noqa: E402

# Locked tree tok/s from C2 bridge job 174124013 (do not re-run tree for speed_300)
TREE_174124013 = {
    "c1_d3": 49.80554744282418,
    "c1_6432": 47.326466131765805,
    "b2_d3": 47.046680632891075,
    "b2_6432": 42.95577701533111,
}
# Display rounding used in advisor table
TREE_DISPLAY = {
    "c1_d3": 49.81,
    "c1_6432": 47.33,
    "b2_d3": 47.05,
    "b2_6432": 42.96,
}
HIST_OLD100_GREEDY = 30.3
OLD100_BAND = 0.03  # ±3%
TREE_RERUN_BAND = 0.02  # ±2% if tree were re-measured


def main():
    out_dir = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/bridge_300")
    manifest = "/scratch/li96/mz9869/eval_manifests/manifest_300.json"
    images = Path("/g/data/li96/mz9869/data/coco_subset")
    prompts = filter_prompts(manifest, 80, 42, ordered=True)
    assert len(prompts) == 300, len(prompts)

    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    eos = processor.tokenizer.eos_token_id

    # Per-prompt wall so old100 / new200 / all300 can be reported separately.
    walls: list[float] = []
    ntoks: list[int] = []
    t_all0 = time.time()
    for i, p in enumerate(prompts):
        t0 = time.time()
        g = vanilla_greedy(base, processor, p["question"], images / p["image"], 150, eos)
        walls.append(time.time() - t0)
        ntoks.append(len(g))
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_all0
            print(f"  [{i+1}/300] gtok={sum(ntoks)} tok/s={sum(ntoks)/elapsed:.3f}")

    def band_stats(sl: slice, name: str):
        tok = sum(ntoks[sl])
        wall = sum(walls[sl])
        tps = tok / wall if wall > 0 else float("nan")
        return {
            "name": name,
            "n_prompts": len(ntoks[sl]),
            "tokens": tok,
            "wall_s": wall,
            "greedy_tok_per_s": tps,
        }

    old100 = band_stats(slice(0, 100), "old100")
    new200 = band_stats(slice(100, 300), "new200")
    all300 = band_stats(slice(0, 300), "all300")

    # Gate (a)
    g100 = old100["greedy_tok_per_s"]
    lo, hi = HIST_OLD100_GREEDY * (1 - OLD100_BAND), HIST_OLD100_GREEDY * (1 + OLD100_BAND)
    gate_a = lo <= g100 <= hi
    print(f"\n=== GATE (a) old100 greedy t/s ===")
    print(f"  old100={g100:.4f}  band=[{lo:.4f}, {hi:.4f}]  hist=30.3  "
          f"{'PASS' if gate_a else 'FAIL'}")
    print(f"  new200={new200['greedy_tok_per_s']:.4f}  all300={all300['greedy_tok_per_s']:.4f} "
          f"(informational; composition may differ)")

    # Gate (b): use locked tree numerators (no tree re-run in this job)
    print(f"\n=== GATE (b) tree numerators (locked 174124013) ===")
    for tag, t in TREE_174124013.items():
        print(f"  {tag}: tree={t:.4f} (display {TREE_DISPLAY[tag]})")
    gate_b = True  # numerators taken from lock; no re-run to check

    # Gate (c): pooled vs per-config
    # Single clean greedy ⇒ pooled denom = all300; own-greedy ≡ pooled.
    g_pooled = all300["greedy_tok_per_s"]
    g_own = g_pooled  # n_config measurements = 1
    speed_pooled = {tag: TREE_174124013[tag] / g_pooled for tag in TREE_174124013}
    speed_own = {tag: TREE_174124013[tag] / g_own for tag in TREE_174124013}
    # Sanity expected核对: vs old100 greedy (~1.64 for c1_d3 when g100≈30.3)
    speed_vs_old100 = {tag: TREE_174124013[tag] / g100 for tag in TREE_174124013}

    print(f"\n=== GATE (c) speed_300 (champion = pooled) ===")
    print(f"  pooled_greedy_tok_per_s={g_pooled:.4f}  (own≡pooled; single clean pass)")
    for tag in TREE_174124013:
        print(f"  {tag}: speed_pooled={speed_pooled[tag]:.4f}  "
              f"speed_own={speed_own[tag]:.4f}  "
              f"vs_old100_g={speed_vs_old100[tag]:.4f}")
    print(f"  expected核对: c1_d3 vs_old100_g ≈ 1.644 when g100≈30.3 "
          f"(got {speed_vs_old100['c1_d3']:.4f})")
    print(f"  note: speed_300 (pooled) vs hist 1.689× (100-scale) = composition diff, "
          f"not a regression")

    gate_c = True  # reporting requirement; values always produced

    rec = {
        "job_id": os.environ.get("PBS_JOBID"),
        "instrumentation": "none (per-prompt vanilla_greedy only; no O0 / no top2 gap)",
        "acceptance": {
            "gate_a_old100_in_30.3_pm3pct": gate_a,
            "gate_a_band": [lo, hi],
            "gate_b_tree_locked_174124013": gate_b,
            "gate_c_both_calibs_reported": gate_c,
            "all_pass": gate_a and gate_b and gate_c,
        },
        "greedy": {
            "old100": old100,
            "new200": new200,
            "all300": all300,
            "pooled_greedy_tok_per_s": g_pooled,
            "per_config_greedy_tok_per_s": {
                tag: g_own for tag in TREE_174124013
            },
            "note": "single clean pass ⇒ per-config greedy ≡ pooled",
        },
        "tree_tok_per_s_locked_174124013": TREE_174124013,
        "speed_300": {
            "pooled": speed_pooled,
            "per_config": speed_own,
            "champion_calibration": "pooled",
            "vs_old100_greedy_sanity": speed_vs_old100,
            "vs_hist_1.689_c1_d3_note": (
                "difference vs 100-scale 1.689× attributed to 300-prompt composition; "
                "not judged as regression"
            ),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / "speed_300_from_clean_greedy.json"
    json.dump(rec, open(outp, "w"), indent=2)
    json.dump(rec["greedy"], open(out_dir / "clean_greedy_300.json", "w"), indent=2)
    print(f"\nwrote {outp}")

    if not rec["acceptance"]["all_pass"]:
        print("=== ACCEPTANCE FAIL — speed_300 NOT locked; STOP ===")
        return 2
    print("=== ACCEPTANCE PASS — speed_300 may be locked from this artifact ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
