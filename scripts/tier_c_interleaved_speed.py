"""Tier C — interleaved e2e speed final (v2_hardcap runner).

Same process / same model instance. Per prompt, randomly order
{static_d3, dyn_n24, greedy}; ≥3 independent blocks; warmup then time.

Each method/block stores a stable SHA-256 of emitted token IDs.
Primary analysis: byte-identical subset (all three methods agree on tokens
within every block, and hashes stable across blocks). Sensitivity: complement.

Pre-registered judgment (primary subset):
  Let d_i = speedup(dyn_n24)_i − speedup(static_d3)_i
  with speedup = greedy_wall / method_wall (within-block walls).
  Walls first averaged across blocks per (prompt, method).
  95% CI = mean ± 1.96·SE. 1% band = ±0.01 · mean(static_d3 speedup).
  CI fully outside band → speed champion = dynamic (if mean>0) or static.
  CI intersects band → "并列/未分".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decode.common import (  # noqa: E402
    cfg_attr,
    filter_prompts,
    load_base,
    load_head,
    vanilla_greedy,
)
from scripts.eval_acceptance_tree import (  # noqa: E402
    _cuda_sync,
    _e2e_wall_call,
    run_one_prompt_tree_folded,
)

METHODS = ("static_d3", "dyn_n24", "greedy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-prompts", type=int, default=300)
    p.add_argument("--min-ref-words", type=int, default=80)
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    return p.parse_args()


def token_seq_hash(ids: Sequence[int]) -> str:
    h = hashlib.sha256()
    for t in ids:
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def _gpu_meta() -> Dict:
    meta: Dict = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "pbs_jobid": os.environ.get("PBS_JOBID"),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        meta.update({
            "device_index": idx,
            "device_name": torch.cuda.get_device_name(idx),
            "total_memory_bytes": int(props.total_memory),
            "allocated_bytes": int(torch.cuda.memory_allocated(idx)),
            "reserved_bytes": int(torch.cuda.memory_reserved(idx)),
        })
    return meta


def _run_method(name: str, base, head, processor, prompt, images_dir, max_new, eos_id):
    if name == "greedy":
        toks = vanilla_greedy(
            base, processor, prompt["question"], images_dir / prompt["image"],
            max_new, eos_id,
        )
        return {
            "n_tokens": len(toks),
            "token_hash": token_seq_hash(toks),
            "hit_eos": bool(toks and toks[-1] == eos_id),
        }
    if name == "static_d3":
        r = run_one_prompt_tree_folded(
            base, head, processor, prompt, images_dir, max_new, eos_id,
            fanout=[1, 3, 2, 1, 0], max_nodes=16, profile=False,
            depth1_floor=True, tree_builder="static",
        )
    elif name == "dyn_n24":
        r = run_one_prompt_tree_folded(
            base, head, processor, prompt, images_dir, max_new, eos_id,
            fanout=[1, 8, 8, 8, 8], max_nodes=24, profile=False,
            depth1_floor=True, tree_builder="dynamic",
        )
    else:
        raise ValueError(name)
    toks = r["emitted_tokens"]
    return {
        "n_tokens": int(r["total_emitted"]),
        "token_hash": token_seq_hash(toks),
        "hit_eos": bool(r["hit_eos"]),
        "sigma": float(r["sigma"]),
        "mean_width": float(r["mean_width"]),
        "rounds": int(r["rounds"]),
    }


def _ci_verdict(diffs: List[float], spd_static: List[float]) -> Dict:
    dm = statistics.mean(diffs)
    dse = statistics.stdev(diffs) / math.sqrt(len(diffs))
    z = 1.96
    ci_lo, ci_hi = dm - z * dse, dm + z * dse
    anchor = statistics.mean(spd_static)
    band = 0.01 * anchor
    fully_outside = (ci_hi < -band) or (ci_lo > band)
    if fully_outside:
        verdict = (
            "speed champion = dyn_k8_n24" if dm > 0 else "speed champion = static_c1_d3"
        )
    else:
        verdict = "并列/未分"
    return {
        "delta_mean": dm,
        "delta_se": dse,
        "n": len(diffs),
        "ci95": [ci_lo, ci_hi],
        "anchor_static_speedup": anchor,
        "band_1pct": band,
        "band_interval": [-band, band],
        "ci_fully_outside_band": fully_outside,
        "verdict": verdict,
    }


def main() -> int:
    args = parse_args()
    for k, v in vars(args).items():
        print(f"[args] {k} = {v}")

    base, processor = load_base(args.model_id)
    head = load_head(args.ckpt, cfg_attr(base.config, "hidden_size"),
                     cfg_attr(base.config, "vocab_size"))
    eos_id = processor.tokenizer.eos_token_id
    images_dir = Path(args.images_dir)
    prompts = filter_prompts(
        args.manifest, args.min_ref_words, args.seed, ordered=True,
    )[: args.n_prompts]
    print(f"[data] {len(prompts)} prompts; blocks={args.n_blocks}")

    gpu0 = _gpu_meta()
    print(f"[gpu] {gpu0.get('device_name')} mem={gpu0.get('total_memory_bytes')}")

    print("[warmup] prompt 0 × all methods…")
    for m in METHODS:
        _run_method(m, base, head, processor, prompts[0], images_dir,
                    args.max_new_tokens, eos_id)
    _cuda_sync()

    raw_blocks: List[Dict] = []
    t_job0 = time.time()
    for b in range(args.n_blocks):
        rng = torch.Generator()
        rng.manual_seed(args.seed + 10_000 * (b + 1))
        block_rows: List[Dict] = []
        print(f"\n===== BLOCK {b+1}/{args.n_blocks} =====")
        for i, p in enumerate(prompts):
            order_idx = torch.randperm(len(METHODS), generator=rng).tolist()
            order = [METHODS[j] for j in order_idx]
            runs: Dict[str, Dict] = {}
            for m in order:
                meta, wall = _e2e_wall_call(
                    lambda m=m, p=p: _run_method(
                        m, base, head, processor, p, images_dir,
                        args.max_new_tokens, eos_id,
                    )
                )
                runs[m] = {**meta, "wall_s": float(wall)}
            row = {
                "id": p.get("id", i),
                "prompt_index": i,
                "block": b,
                "order": order,
                "runs": runs,
                "gpu": {
                    "allocated_bytes": int(torch.cuda.memory_allocated())
                    if torch.cuda.is_available() else None,
                    "reserved_bytes": int(torch.cuda.memory_reserved())
                    if torch.cuda.is_available() else None,
                },
            }
            gw = runs["greedy"]["wall_s"]
            for m in ("static_d3", "dyn_n24"):
                sw = runs[m]["wall_s"]
                row[f"speedup_{m}"] = (gw / sw) if sw > 0 else None
            row["paired_delta_dyn_minus_static"] = (
                row["speedup_dyn_n24"] - row["speedup_static_d3"]
                if row["speedup_dyn_n24"] is not None
                and row["speedup_static_d3"] is not None
                else None
            )
            row["byte_identical_in_block"] = (
                runs["static_d3"]["token_hash"]
                == runs["dyn_n24"]["token_hash"]
                == runs["greedy"]["token_hash"]
            )
            block_rows.append(row)
            if (i + 1) % 25 == 0 or i == 0:
                print(
                    f"[b{b+1} {i+1:>3}/{len(prompts)}] order={order} "
                    f"Δspd={row['paired_delta_dyn_minus_static']:+.4f} "
                    f"ident={row['byte_identical_in_block']} g={gw:.2f}s"
                )
        raw_blocks.append({"block": b, "seed": args.seed + 10_000 * (b + 1),
                           "per_prompt": block_rows})

    n = len(prompts)
    nb = float(args.n_blocks)
    avg_wall = {m: [0.0] * n for m in METHODS}
    avg_tok = {m: [0.0] * n for m in METHODS}
    for blk in raw_blocks:
        for row in blk["per_prompt"]:
            i = row["prompt_index"]
            for m in METHODS:
                avg_wall[m][i] += row["runs"][m]["wall_s"]
                avg_tok[m][i] += row["runs"][m]["n_tokens"]
    for m in METHODS:
        avg_wall[m] = [w / nb for w in avg_wall[m]]
        avg_tok[m] = [t / nb for t in avg_tok[m]]

    # Byte-identical subset: agree across methods in every block + stable hashes
    primary_idx: List[int] = []
    sensitivity_idx: List[int] = []
    for i in range(n):
        hashes_by_method = {m: [] for m in METHODS}
        ok = True
        for blk in raw_blocks:
            row = blk["per_prompt"][i]
            if not row["byte_identical_in_block"]:
                ok = False
            for m in METHODS:
                hashes_by_method[m].append(row["runs"][m]["token_hash"])
        for m in METHODS:
            if len(set(hashes_by_method[m])) != 1:
                ok = False
        (primary_idx if ok else sensitivity_idx).append(i)

    def _subset_stats(indices: List[int]) -> Optional[Dict]:
        if len(indices) < 2:
            return None
        diffs, spd_s, spd_d = [], [], []
        paired = []
        for i in indices:
            gw, ss, sd = avg_wall["greedy"][i], avg_wall["static_d3"][i], avg_wall["dyn_n24"][i]
            su_s = gw / ss if ss > 0 else None
            su_d = gw / sd if sd > 0 else None
            dlt = (su_d - su_s) if (su_d is not None and su_s is not None) else None
            paired.append({
                "id": prompts[i].get("id", i),
                "prompt_index": i,
                "wall_greedy_s": gw,
                "wall_static_d3_s": ss,
                "wall_dyn_n24_s": sd,
                "tokens_greedy": avg_tok["greedy"][i],
                "tokens_static_d3": avg_tok["static_d3"][i],
                "tokens_dyn_n24": avg_tok["dyn_n24"][i],
                "speedup_static_d3": su_s,
                "speedup_dyn_n24": su_d,
                "paired_delta_dyn_minus_static": dlt,
            })
            if dlt is not None:
                diffs.append(dlt)
                spd_s.append(su_s)
                spd_d.append(su_d)
        v = _ci_verdict(diffs, spd_s)
        return {
            "n": len(indices),
            "static_d3_speedup_mean": statistics.mean(spd_s),
            "static_d3_speedup_se": statistics.stdev(spd_s) / math.sqrt(len(spd_s)),
            "dyn_n24_speedup_mean": statistics.mean(spd_d),
            "dyn_n24_speedup_se": statistics.stdev(spd_d) / math.sqrt(len(spd_d)),
            "dyn_n24_vs_static_d3": v,
            "per_prompt": paired,
        }

    primary = _subset_stats(primary_idx)
    sensitivity = _subset_stats(sensitivity_idx)
    all_stats = _subset_stats(list(range(n)))

    per_block = []
    z = 1.96
    for blk in raw_blocks:
        bd = [r["paired_delta_dyn_minus_static"] for r in blk["per_prompt"]
              if r["paired_delta_dyn_minus_static"] is not None]
        bm = statistics.mean(bd)
        bse = statistics.stdev(bd) / math.sqrt(len(bd))
        per_block.append({
            "block": blk["block"], "delta_mean": bm, "delta_se": bse, "n": len(bd),
            "ci95": [bm - z * bse, bm + z * bse],
        })

    inconclusive = len(primary_idx) < 2
    if inconclusive:
        primary_v = {
            "verdict": "INCONCLUSIVE",
            "reason": f"primary byte-identical n={len(primary_idx)} < 2",
            "n": len(primary_idx),
        }
        official_static = official_dyn = official_v = None
    else:
        primary_v = primary["dyn_n24_vs_static_d3"]
        official_static = primary["static_d3_speedup_mean"]
        official_dyn = primary["dyn_n24_speedup_mean"]
        official_v = primary_v

    summary = {
        "protocol": "tier_c_interleaved_e2e_wall",
        "runner": "v2_hardcap",
        "tier": "D_token_hash",
        "job": os.environ.get("PBS_JOBID"),
        "n_prompts": n,
        "n_blocks": args.n_blocks,
        "seed": args.seed,
        "methods": list(METHODS),
        "gpu_start": gpu0,
        "gpu_end": _gpu_meta(),
        "wall_job_s": time.time() - t_job0,
        "n_byte_identical_primary": len(primary_idx),
        "n_sensitivity": len(sensitivity_idx),
        "primary_byte_identical": (
            {k: primary[k] for k in primary if k != "per_prompt"}
            if primary else None
        ),
        "sensitivity_non_identical": (
            {k: sensitivity[k] for k in sensitivity if k != "per_prompt"}
            if sensitivity else None
        ),
        "all_prompts": (
            {k: all_stats[k] for k in all_stats if k != "per_prompt"}
            if all_stats else None
        ),
        # Official fields = primary only; no all-prompts fallback
        "inconclusive": inconclusive,
        "static_d3_speedup_mean": official_static,
        "static_d3_speedup_se": None if inconclusive else primary["static_d3_speedup_se"],
        "dyn_n24_speedup_mean": official_dyn,
        "dyn_n24_speedup_se": None if inconclusive else primary["dyn_n24_speedup_se"],
        "dyn_n24_vs_static_d3": official_v if official_v is not None else primary_v,
        "per_block": per_block,
        "pre_reg": (
            "95% CI of paired (dyn_n24−static_d3) on byte-identical primary; "
            "fully outside ±1%·anchor → speed champion; else 并列/未分; "
            "primary n<2 → INCONCLUSIVE (no all-prompts fallback)"
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out.with_name(out.stem + "_raw.json")
    paired_path = out.with_name(out.stem + "_paired.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"blocks": raw_blocks, "gpu_start": gpu0,
                   "primary_indices": primary_idx,
                   "sensitivity_indices": sensitivity_idx}, f)
    with open(paired_path, "w", encoding="utf-8") as f:
        json.dump({
            "primary_per_prompt": (primary or {}).get("per_prompt"),
            "sensitivity_per_prompt": (sensitivity or {}).get("per_prompt"),
            "all_per_prompt": (all_stats or {}).get("per_prompt"),
        }, f, indent=2, ensure_ascii=False)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    v = summary["dyn_n24_vs_static_d3"]
    print("\n===== TIER C VERDICT (primary byte-identical) =====")
    print(f"primary n={len(primary_idx)}  sensitivity n={len(sensitivity_idx)}")
    if inconclusive:
        print(f"VERDICT: INCONCLUSIVE (primary n={len(primary_idx)} < 2)")
    else:
        print(f"static_d3 spd = {summary['static_d3_speedup_mean']:.4f} ± "
              f"{summary['static_d3_speedup_se']:.4f}")
        print(f"dyn_n24  spd = {summary['dyn_n24_speedup_mean']:.4f} ± "
              f"{summary['dyn_n24_speedup_se']:.4f}")
        print(f"Δ(dyn−static) = {v['delta_mean']:+.4f} ± {v['delta_se']:.4f}  "
              f"95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
        print(f"1% band = ±{v['band_1pct']:.4f}  fully_outside={v['ci_fully_outside_band']}")
        print(f"VERDICT: {v['verdict']}")
    if sensitivity and "dyn_n24_vs_static_d3" in sensitivity:
        sv = sensitivity["dyn_n24_vs_static_d3"]
        print(f"[sensitivity n={sv['n']}] Δ={sv['delta_mean']:+.4f} "
              f"CI={sv['ci95']} verdict={sv['verdict']}")
    if all_stats and "dyn_n24_vs_static_d3" in all_stats:
        av = all_stats["dyn_n24_vs_static_d3"]
        print(f"[all-prompts n={av['n']}] Δ={av['delta_mean']:+.4f} "
              f"CI={av['ci95']} verdict={av['verdict']} (report-only)")
    print(f"wrote {out}\nwrote {raw_path}\nwrote {paired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
