#!/usr/bin/env python3
"""Candidate-specific re-probe of recorded O0 near_ties (512).

For each near_tie row: greedy to first-diverge pos via hook path, dump h,
CPU lm_head.float() @ h → full-precision logits; reclassify with the
gap-only rule 0 ≤ logit[greedy_top1] − logit[spec_tok] ≤ 0.15 (Round-7).
``spec_rank`` and ``spec_tok == greedy_top2`` are recorded as diagnostics
only. One V100 PBS job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from decode.common import (  # noqa: E402
    O0_KERNEL_BAND,
    apply_max_pixels,
    candidate_probe_from_logits,
    classify_o0_vs_ref,
    filter_prompts,
    greedy_dump_h_at,
    is_candidate_near_tie,
    load_base,
    mask_phantom_,
)

C1 = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle")

# MM-Vet runs were launched with --max-pixels 501760; the probe must reproduce
# the exact same vision-token budget or the replayed context is a different one.
OOD_MAX_PIXELS = 501760

# tag → (layout, ordered_manifest, min_ref_words, max_pixels)
TAG_LAYOUT = {
    "gate100_c1_d3": ("100", False, 80, None),
    "gate100_c1_6432": ("100", False, 80, None),
    "gate100_b2_wide": ("100", False, 80, None),
    "bridge300_c1_d3": ("300", True, 80, None),
    "bridge300_c1_6432": ("300", True, 80, None),
    "bridge300_b2_d3": ("300", True, 80, None),
    "bridge300_b2_6432": ("300", True, 80, None),
    "dyn_k8_n24": ("300", True, 80, None),
    "dyn_k8_n32": ("300", True, 80, None),
    "O1_c1_d3": ("ood", True, 0, OOD_MAX_PIXELS),
    "O2_c1_6432": ("ood", True, 0, OOD_MAX_PIXELS),
    "O3_b2_d3": ("ood", True, 0, OOD_MAX_PIXELS),
    "O4_b2_6432": ("ood", True, 0, OOD_MAX_PIXELS),
}


def collect_near_ties(reports_dir: Path) -> List[Dict]:
    rows = []
    for tag in TAG_LAYOUT:
        p = reports_dir / f"{tag}.o0_report.json"
        if not p.exists():
            raise FileNotFoundError(p)
        rep = json.loads(p.read_text(encoding="utf-8"))
        for nt in rep.get("near_ties") or []:
            rows.append({**nt, "tag": tag})
    # group by layout so the processor's max_pixels is set once per group
    rows.sort(key=lambda r: (TAG_LAYOUT[r["tag"]][0], r["tag"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default=str(C1 / "tier_d_repro_fullcover"))
    ap.add_argument("--out-dir", default=str(C1 / "round6_candidate_reprobe"))
    ap.add_argument("--manifest-100",
                    default=os.environ.get(
                        "MEDUSA_MANIFEST100",
                        "/g/data/li96/mz9869/data/llava_subset_2k.json"))
    ap.add_argument("--manifest-300",
                    default=os.environ.get(
                        "MEDUSA_MANIFEST300",
                        "/scratch/li96/mz9869/eval_manifests/manifest_300.json"))
    ap.add_argument("--manifest-ood",
                    default=os.environ.get(
                        "MEDUSA_MANIFEST_OOD",
                        "/scratch/li96/mz9869/ood_eval/mmvet/manifest_mmvet_218.json"))
    ap.add_argument("--images",
                    default=os.environ.get(
                        "MEDUSA_IMAGES", "/g/data/li96/mz9869/data/coco_subset"))
    ap.add_argument("--images-ood",
                    default=os.environ.get(
                        "MEDUSA_IMAGES_OOD",
                        "/scratch/li96/mz9869/ood_eval/mmvet/mm-vet/images"))
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-items", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = collect_near_ties(Path(args.reports_dir))
    if args.max_items:
        items = items[: args.max_items]
    print(f"[reprobe] n_items={len(items)} band={O0_KERNEL_BAND} "
          f"job={os.environ.get('PBS_JOBID')}")

    print("[reprobe] loading base…")
    base, processor = load_base(args.model_id)
    eos = processor.tokenizer.eos_token_id
    lm_w = base.lm_head.weight.detach().float().cpu()
    device = "cuda:0"

    # prompt caches keyed by layout
    prompt_cache: Dict[str, Dict[str, Dict]] = {}

    def prompts_for(tag: str) -> Dict[str, Dict]:
        layout, ordered, min_ref, _ = TAG_LAYOUT[tag]
        key = f"{layout}|{ordered}|{min_ref}"
        if key not in prompt_cache:
            if layout == "100":
                man, imgs, n = args.manifest_100, args.images, 100
            elif layout == "300":
                man, imgs, n = args.manifest_300, args.images, 300
            else:
                man, imgs, n = args.manifest_ood, args.images_ood, 218
            ps = filter_prompts(man, min_ref, 42, ordered=ordered)[:n]
            prompt_cache[key] = {
                "_images": imgs,
                **{str(p["id"]): p for p in ps},
            }
        return prompt_cache[key]

    default_ip_size = getattr(processor.image_processor, "size", None)
    default_max_pixels = getattr(processor.image_processor, "max_pixels", None)
    current_max_pixels = "__unset__"

    def set_max_pixels(mp) -> None:
        """Match the vision-token budget of the run that produced the near_tie."""
        nonlocal current_max_pixels
        if mp == current_max_pixels:
            return
        if mp is None:
            if default_max_pixels is not None:
                processor.image_processor.max_pixels = default_max_pixels
            if default_ip_size is not None:
                processor.image_processor.size = default_ip_size
            processor.max_pixels = default_max_pixels
        else:
            apply_max_pixels(processor, mp)
        current_max_pixels = mp
        print(f"[reprobe] max_pixels -> {mp} "
              f"(ip.size={getattr(processor.image_processor, 'size', None)})")

    results = []
    n_pass = n_fail = n_err = n_prefix_bad = 0
    for i, nt in enumerate(items):
        tag = nt["tag"]
        pid = str(nt["prompt_id"])
        pos = int(nt["pos"])
        spec_tok = nt["spec_tok"]
        greedy_tok = nt.get("greedy_tok")
        max_pixels = TAG_LAYOUT[tag][3]
        pc = prompts_for(tag)
        p = pc.get(pid)
        images_dir = Path(pc["_images"])
        if p is None:
            rec = {**nt, "status": "MISSING_PROMPT", "new_kind": "hard",
                   "candidate_pass": False}
            results.append(rec)
            n_err += 1
            print(f"[{i+1}/{len(items)}] MISSING prompt {tag}/{pid}")
            continue
        try:
            set_max_pixels(max_pixels)
            h, emitted, mem = greedy_dump_h_at(
                base, processor, p["question"], images_dir / p["image"],
                args.max_new_tokens, eos, pos, device=device,
            )
            # Context fidelity: replayed greedy prefix must reproduce the
            # recorded greedy context (independent of the classification).
            # The recorded context is only the last ≤5 tokens, so this check is
            # a tail check, not whole-prefix equality — named accordingly.
            want_before = list((nt.get("greedy_context") or {}).get("before") or [])
            got_before = list(emitted[-len(want_before):]) if want_before else []
            prefix_ok = (got_before == want_before)
            replayed_prefix = [int(t) for t in emitted]
            replayed_sha = hashlib.sha256(
                ",".join(str(t) for t in replayed_prefix).encode("utf-8")
            ).hexdigest()
            logits = mask_phantom_(torch.nn.functional.linear(h.float(), lm_w))
            probe = candidate_probe_from_logits(logits, spec_tok)
            # Synthetic sequences for classifier (only mid-diverge matters)
            # Build minimal mid-diverge pair at pos=0 for classify API
            spec_seq = [int(spec_tok)]
            ref_seq = [int(greedy_tok) if greedy_tok is not None else probe["greedy_top1"]]
            cls = classify_o0_vs_ref(
                spec_seq, ref_seq,
                top2_gap=probe["top2_logit_gap"],
                greedy_top1=probe["greedy_top1"],
                greedy_top2=probe["greedy_top2"],
                gap_spec=probe["gap_spec"],
                spec_rank=probe["spec_rank"],
            )
            passed = cls["kind"] == "near_tie"
            # Double-check predicate
            assert passed == is_candidate_near_tie(
                spec_tok, probe["greedy_top2"], probe["gap_spec"])
            # A row only counts once the replayed context is provably the
            # recorded one; otherwise it is INCONCLUSIVE (never silently hard).
            status = "OK" if prefix_ok else "PREFIX_MISMATCH"
            rec = {
                "tag": tag,
                "prompt_id": pid,
                "image": p["image"],
                "pos": pos,
                "max_pixels": max_pixels,
                "spec_tok": int(spec_tok),
                "greedy_tok_recorded": greedy_tok,
                "tail_context_expected": want_before,
                "tail_context_replayed": got_before,
                "tail_context_len": len(want_before),
                "prefix_ok": prefix_ok,
                "replayed_prefix_len": len(replayed_prefix),
                "replayed_prefix_sha256": replayed_sha,
                "status": status,
                "old_top2_logit_gap": nt.get("top2_logit_gap"),
                **probe,
                "new_kind": cls["kind"] if prefix_ok else "INCONCLUSIVE",
                "candidate_pass": bool(passed and prefix_ok),
                "method": "greedy_dump_h_at + lm_head.float()@h CPU",
                "peak_alloc_gib": mem.get("peak_alloc_gib"),
            }
            results.append(rec)
            if not prefix_ok:
                n_prefix_bad += 1
            elif passed:
                n_pass += 1
            else:
                n_fail += 1
            print(
                f"[{i+1}/{len(items)}] {tag}/{pid} pos={pos} "
                f"rank={probe['spec_rank']} gap_spec={probe['gap_spec']:.4f} "
                f"top2={probe['greedy_top2']} spec={spec_tok} → "
                f"{rec['new_kind']}{'' if prefix_ok else ' [PREFIX_MISMATCH]'}"
            )
        except Exception as e:
            results.append({
                **{k: nt.get(k) for k in ("tag", "prompt_id", "pos", "spec_tok")},
                "status": "ERROR", "error": repr(e), "new_kind": "hard",
                "candidate_pass": False,
            })
            n_err += 1
            print(f"[{i+1}/{len(items)}] ERROR {tag}/{pid}: {e!r}")
        if (i + 1) % 25 == 0:
            # checkpoint
            (out_dir / "reprobe_rows_partial.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    ok_rows = [r for r in results if r.get("prefix_ok")]
    ranks = [r["spec_rank"] for r in ok_rows if r.get("spec_rank") is not None]
    gaps = [r["gap_spec"] for r in ok_rows if r.get("gap_spec") is not None]
    kind_c = Counter(r.get("new_kind") for r in results)
    summary = {
        "job": os.environ.get("PBS_JOBID"),
        "protocol": "candidate_specific_near_tie_reprobe",
        "near_tie_rule": "candidate_specific_gap_only",
        "band": O0_KERNEL_BAND,
        "n_total": len(results),
        # Tail check only: the recorded greedy_context holds ≤5 tokens.
        "n_tail_context_verified": len(ok_rows),
        "tail_context_window": "last_up_to_5_tokens",
        "n_pass_near_tie": n_pass,
        "n_fail_hard": n_fail,
        "n_prefix_mismatch": n_prefix_bad,
        "n_error": n_err,
        "all_pass": (
            n_pass == len(results)
            and n_err == 0 and n_fail == 0 and n_prefix_bad == 0
        ),
        "kind_counts": dict(kind_c),
        "spec_rank_hist": dict(Counter(ranks)),
        "gap_spec": {
            "min": min(gaps) if gaps else None,
            "max": max(gaps) if gaps else None,
            "mean": (sum(gaps) / len(gaps)) if gaps else None,
            "n_le_band": sum(1 for g in gaps if g <= O0_KERNEL_BAND),
            "n_gt_band": sum(1 for g in gaps if g > O0_KERNEL_BAND),
        },
        "n_spec_eq_top2": sum(
            1 for r in ok_rows
            if r.get("spec_tok") is not None
            and r.get("greedy_top2") is not None
            and int(r["spec_tok"]) == int(r["greedy_top2"])
        ),
        "n_recorded_greedy_in_top2": sum(
            1 for r in ok_rows
            if r.get("greedy_tok_recorded") in (r.get("greedy_top1"), r.get("greedy_top2"))
        ),
        "preregistered_read": (
            "all_pass → upgrade safety claim to candidate-specific "
            "(evidence 512/512 + historical item3 10/10); "
            "any fail → hold narrowed claim; review fail rows as hard candidates; "
            "PREFIX_MISMATCH = probe invalid (not divergence evidence)"
        ),
        "fails": [r for r in results if not r.get("candidate_pass")],
    }
    (out_dir / "reprobe_rows.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "reprobe_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Human-readable table
    lines = [
        "# Candidate-specific re-probe (gap-only rule)",
        f"n={summary['n_total']} "
        f"tail_context_verified={summary['n_tail_context_verified']} "
        f"pass={n_pass} fail={n_fail} prefix_mismatch={n_prefix_bad} err={n_err} "
        f"all_pass={summary['all_pass']}",
        f"spec==top2: {summary['n_spec_eq_top2']}  "
        f"recorded_greedy in top2: {summary['n_recorded_greedy_in_top2']}",
        f"gap_spec: min={summary['gap_spec']['min']} max={summary['gap_spec']['max']} "
        f"mean={summary['gap_spec']['mean']} "
        f"<=band={summary['gap_spec']['n_le_band']} >band={summary['gap_spec']['n_gt_band']}",
        f"rank_hist: {summary['spec_rank_hist']}",
        "",
        "tag\tprompt_id\tpos\tspec\ttop1\ttop2\trank\tgap_spec\told_top2_gap\t"
        "prefix_ok\tkind",
    ]
    for r in results:
        lines.append(
            f"{r.get('tag')}\t{r.get('prompt_id')}\t{r.get('pos')}\t"
            f"{r.get('spec_tok')}\t{r.get('greedy_top1')}\t{r.get('greedy_top2')}\t"
            f"{r.get('spec_rank')}\t{r.get('gap_spec')}\t{r.get('old_top2_logit_gap')}\t"
            f"{r.get('prefix_ok')}\t{r.get('new_kind')}"
        )
    (out_dir / "reprobe_table.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "n_total", "n_tail_context_verified", "n_pass_near_tie", "n_fail_hard",
        "n_prefix_mismatch", "n_error", "all_pass", "kind_counts", "gap_spec",
        "n_spec_eq_top2", "n_recorded_greedy_in_top2", "spec_rank_hist",
    )}, indent=2, ensure_ascii=False))
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
