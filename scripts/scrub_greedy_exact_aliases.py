#!/usr/bin/env python3.11
"""Normalize legacy O0 machine fields in release JSON copies.

- removes forbidden ``greedy_exact_*`` / ``o0_greedy_exact*`` aliases
  (sole allowed legacy name: ``legacy_numerical_safety_pass_deprecated``)
- renames ``o0_spec_pass`` → ``o0_archive_reproducibility_pass``
- marks superseded ``official_claim`` strings as retracted and points at the
  current gap-only claim + its re-verification, without deleting what the
  original run recorded
- surfaces ``n_context_verified`` under its accurate name
  ``n_tail_context_verified`` (the recorded context is only the last ≤5 tokens)

Anything under an ``artifacts/provenance/`` directory is left byte-identical:
those files are kept precisely because they show what a superseded run emitted.

All IO: encoding=utf-8.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SKIP_NAMES = {
    "FILE_MANIFEST.json",
    "FILE_MANIFEST.sha256",
    "INTEGRITY_CHECK.json",
    "SUBMIT_STAMP.json",
    "COVER_CHECK.json",
}

# Substrings identifying a claim string that no longer holds:
#   "hard/material" — pre-Round-6 wording
#   "spec_tok == greedy_top2 or" — Round-6 OR-shortcut, a spec error
RETRACTED_CLAIM_MARKERS = ("hard/material", "spec_tok == greedy_top2 or")
CURRENT_CLAIM = (
    "algorithmic greedy lossless; every first mid-sequence divergence is a "
    "candidate-specific near_tie "
    "(0 <= logit[greedy_top1]-logit[spec_tok] <= 0.15)"
)
REVERIFY_JOB = "175813855.gadi-pbs"
REVERIFY_RECLASS = "round7_gap_only_reclass (offline strict re-scoring, no GPU rerun)"
PROVENANCE_DIR = "artifacts/provenance"


def is_forbidden_key(k: str) -> bool:
    if k in {
        "greedy_exact_pass",
        "greedy_exact_fails",
        "o0_greedy_exact_pass",
        "greedy_exact_triggers_fail",
    }:
        return True
    if k.startswith("greedy_exact_") or k.startswith("o0_greedy_exact"):
        return True
    return False


def scrub(obj) -> int:
    changed = 0
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if is_forbidden_key(k):
                del obj[k]
                changed += 1
            else:
                changed += scrub(obj[k])
        if obj.get("verdict") == "greedy_exact_equivalence":
            obj["verdict"] = "greedy_numerical_safety"
            changed += 1
        if "o0_spec_pass" in obj:
            val = obj.pop("o0_spec_pass")
            obj.setdefault("o0_archive_reproducibility_pass", val)
            changed += 1
        claim = obj.get("official_claim")
        if isinstance(claim, str) and any(m in claim for m in RETRACTED_CLAIM_MARKERS):
            del obj["official_claim"]
            obj["official_claim_at_run_time_retracted"] = claim
            obj["near_tie_rule_at_run_time"] = (
                "round6_or_shortcut" if "greedy_top2 or" in claim
                else "legacy_top1_top2_gap_only"
            )
            obj["official_claim_current"] = CURRENT_CLAIM
            obj["candidate_specific_reverification_job"] = REVERIFY_JOB
            obj["gap_only_reverification"] = REVERIFY_RECLASS
            changed += 1
        if "n_context_verified" in obj and "n_tail_context_verified" not in obj:
            obj["n_tail_context_verified"] = obj.pop("n_context_verified")
            obj["tail_context_window"] = "last_up_to_5_tokens"
            changed += 1
        if (
            "greedy_numerical_safety_pass" in obj
            and "legacy_numerical_safety_pass_deprecated" not in obj
        ):
            obj["legacy_numerical_safety_pass_deprecated"] = obj[
                "greedy_numerical_safety_pass"
            ]
            changed += 1
    elif isinstance(obj, list):
        for x in obj:
            changed += scrub(x)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    n_files = n_chg = 0
    for p in root.rglob("*.json"):
        if p.name in SKIP_NAMES:
            continue
        if PROVENANCE_DIR in p.relative_to(root).as_posix():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        c = scrub(data)
        if c:
            p.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            n_files += 1
            n_chg += c
    print(json.dumps({"scrubbed_files": n_files, "edits": n_chg}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
