#!/usr/bin/env python3.11
"""Offline strict re-classification of the 512 re-probe rows under gap-only.

No GPU, no re-probe: the Round-6 job (175813855) already recorded, per row,
the full-precision `gap_spec`, `spec_rank`, `greedy_top1/top2` and the tail
context verdict. Round-7 only changes the *predicate* (the `spec_tok ==
greedy_top2` disjunct is dropped), so the stored raw rows are re-scored in
place through the same `classify_o0_vs_ref` the evaluator uses.

Rows whose tail context did not verify stay INCONCLUSIVE — a predicate change
cannot rehabilitate an unverified replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decode.common import (  # noqa: E402
    O0_CLAIM_OFFICIAL,
    O0_KERNEL_BAND,
    classify_o0_vs_ref,
    is_candidate_near_tie,
)

C1 = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle")
OOD_TAGS = {"O1_c1_d3", "O2_c1_6432", "O3_b2_d3", "O4_b2_6432"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reclassify(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        spec_tok = r.get("spec_tok")
        greedy_tok = r.get("greedy_tok_recorded")
        top1 = r.get("greedy_top1")
        gap_spec = r.get("gap_spec")
        tail_ok = bool(r.get("prefix_ok"))
        if spec_tok is None or top1 is None:
            new = {**r, "gap_only_kind": "INCONCLUSIVE", "gap_only_pass": False,
                   "gap_only_note": "row has no probe payload"}
            out.append(new)
            continue
        # Same synthetic mid-diverge pair the probe fed the classifier.
        cls = classify_o0_vs_ref(
            [int(spec_tok)],
            [int(greedy_tok) if greedy_tok is not None else int(top1)],
            top2_gap=r.get("top2_logit_gap"),
            greedy_top1=top1,
            greedy_top2=r.get("greedy_top2"),
            gap_spec=gap_spec,
            spec_rank=r.get("spec_rank"),
        )
        passed = cls["kind"] == "near_tie"
        assert passed == is_candidate_near_tie(
            spec_tok, r.get("greedy_top2"), gap_spec)
        out.append({
            **r,
            "gap_only_kind": cls["kind"] if tail_ok else "INCONCLUSIVE",
            "gap_only_pass": bool(passed and tail_ok),
            "round6_kind": r.get("new_kind"),
            "kind_changed": (cls["kind"] if tail_ok else "INCONCLUSIVE")
            != r.get("new_kind"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(C1 / "round6_candidate_reprobe/reprobe_rows.json"))
    ap.add_argument("--out-dir", default=str(C1 / "round7_gap_only_reclass"))
    ap.add_argument("--source-job", default="175813855.gadi-pbs")
    args = ap.parse_args()

    src = Path(args.rows)
    rows = json.loads(src.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = reclassify(rows)
    ok = [r for r in res if r.get("prefix_ok")]
    gaps = [r["gap_spec"] for r in ok if r.get("gap_spec") is not None]
    ranks = [r["spec_rank"] for r in ok if r.get("spec_rank") is not None]
    n_pass = sum(1 for r in res if r["gap_only_pass"])
    n_hard = sum(1 for r in res if r["gap_only_kind"] == "hard")
    n_incon = sum(1 for r in res if r["gap_only_kind"] == "INCONCLUSIVE")
    changed = [r for r in res if r.get("kind_changed")]
    by_seg = Counter(
        ("ood" if r.get("tag") in OOD_TAGS else "in_domain") for r in res
        if r["gap_only_pass"]
    )

    summary = {
        "protocol": "offline_gap_only_reclassification",
        "source_rows": str(src),
        "source_rows_sha256": sha256_file(src),
        "source_job": args.source_job,
        "gpu_rerun": False,
        "near_tie_rule": "candidate_specific_gap_only",
        "official_claim": O0_CLAIM_OFFICIAL,
        "band": O0_KERNEL_BAND,
        "n_total": len(res),
        "n_tail_context_verified": len(ok),
        "tail_context_window": "last_up_to_5_tokens",
        "n_pass_near_tie": n_pass,
        "n_fail_hard": n_hard,
        "n_inconclusive": n_incon,
        "all_pass": n_pass == len(res) and n_hard == 0 and n_incon == 0,
        "n_kind_changed_vs_round6": len(changed),
        "kind_counts": dict(Counter(r["gap_only_kind"] for r in res)),
        "pass_by_segment": dict(by_seg),
        "spec_rank_hist": dict(Counter(ranks)),
        "gap_spec": {
            "min": min(gaps) if gaps else None,
            "max": max(gaps) if gaps else None,
            "mean": (sum(gaps) / len(gaps)) if gaps else None,
            "n_le_band": sum(1 for g in gaps if g <= O0_KERNEL_BAND),
            "n_gt_band": sum(1 for g in gaps if g > O0_KERNEL_BAND),
            "n_negative": sum(1 for g in gaps if g < 0),
        },
        # Diagnostics retained but not load-bearing under the gap-only rule.
        "diagnostic_n_spec_eq_top2": sum(
            1 for r in ok
            if r.get("greedy_top2") is not None
            and int(r["spec_tok"]) == int(r["greedy_top2"])
        ),
        "diagnostic_n_rank_gt_2_passing": sum(
            1 for r in res
            if r["gap_only_pass"] and (r.get("spec_rank") or 0) > 2
        ),
        "changed_rows": [
            {k: r.get(k) for k in
             ("tag", "prompt_id", "pos", "spec_rank", "gap_spec",
              "round6_kind", "gap_only_kind")}
            for r in changed
        ],
        "fails": [r for r in res if not r["gap_only_pass"]],
    }

    (out_dir / "reclass_rows.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "reclass_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Offline gap-only re-classification of the 512 re-probe rows",
        f"source={src.name} sha256={summary['source_rows_sha256'][:16]}… "
        f"job={args.source_job} gpu_rerun=False",
        f"n={summary['n_total']} tail_context_verified={summary['n_tail_context_verified']} "
        f"pass={n_pass} hard={n_hard} inconclusive={n_incon} "
        f"all_pass={summary['all_pass']} changed_vs_round6={len(changed)}",
        f"gap_spec: min={summary['gap_spec']['min']} max={summary['gap_spec']['max']} "
        f"mean={summary['gap_spec']['mean']} <=band={summary['gap_spec']['n_le_band']} "
        f">band={summary['gap_spec']['n_gt_band']} negative={summary['gap_spec']['n_negative']}",
        f"rank_hist={summary['spec_rank_hist']} "
        f"pass_by_segment={summary['pass_by_segment']}",
        f"diagnostics: spec==top2={summary['diagnostic_n_spec_eq_top2']} "
        f"rank>2 passing={summary['diagnostic_n_rank_gt_2_passing']}",
        "",
        "tag\tprompt_id\tpos\tspec\ttop1\ttop2\trank\tgap_spec\tround6_kind\tgap_only_kind",
    ]
    for r in res:
        lines.append(
            f"{r.get('tag')}\t{r.get('prompt_id')}\t{r.get('pos')}\t"
            f"{r.get('spec_tok')}\t{r.get('greedy_top1')}\t{r.get('greedy_top2')}\t"
            f"{r.get('spec_rank')}\t{r.get('gap_spec')}\t"
            f"{r.get('round6_kind')}\t{r.get('gap_only_kind')}"
        )
    (out_dir / "reclass_table.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:6]))
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
