#!/usr/bin/env python3
"""3b(b/c): offline CPU — feed cand dump to static + dynamic builders; node-set diff.

Patches decode.tree.topk_masked to return the dumped (lp, idx) so both
build_tree_folded and build_tree_folded_dynamic consume the exact live
candidates (no log_softmax renorm). decode/ source is not edited.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import decode.tree as tree_mod  # noqa: E402
from decode.tree import (  # noqa: E402
    build_tree_folded,
    build_tree_folded_dynamic,
    per_depth_widths,
)

VOCAB = 152064

# Set by diff_round before calling builders.
_ACTIVE_LEVELS: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
_ORIG_TOPK = tree_mod.topk_masked


def _patched_topk(logits_1d: torch.Tensor, k: int, max_id: int = 151643):
    """Return dumped candidates; identify head by matching which level tensor we pass.

    Callers pass all_logits[k]; we key by id(logits storage) via a side map set
    before build. Fallback: scan _ACTIVE_LEVELS by object identity of the
    placeholder tensor we embedded in all_logits.
    """
    key = id(logits_1d) if logits_1d.dim() == 1 else id(logits_1d.reshape(-1))
    # builders call .reshape(-1) which may create a view with same storage
    storage_key = logits_1d.untyped_storage().data_ptr()
    for head_k, (lp_full, idx_full, tensor_ref) in _ACTIVE_LEVELS.items():
        if logits_1d.data_ptr() == tensor_ref.data_ptr() or storage_key == tensor_ref.untyped_storage().data_ptr():
            return lp_full[:k].clone(), idx_full[:k].clone()
    # should not happen
    raise RuntimeError(f"patched topk: no dump for tensor ptr={storage_key}")


def node_records(nodes) -> Dict[Tuple[int, ...], dict]:
    by = {n.flat_idx: n for n in nodes}

    def path_of(n):
        toks = []
        cur = n
        while True:
            toks.append(int(cur.token))
            if cur.parent == -1:
                break
            cur = by[cur.parent]
        return tuple(reversed(toks))

    children: Dict[int, List[int]] = {}
    for n in nodes:
        children.setdefault(n.parent, []).append(n.flat_idx)
    slot = {}
    for idxs in children.values():
        for b, fi in enumerate(idxs):
            slot[fi] = b

    out = {}
    for n in nodes:
        p = path_of(n)
        cum = float(n.cum_logprob)
        out[p] = {
            "token": int(n.token),
            "depth": int(n.depth),
            "parent_path": list(p[:-1]) if len(p) > 1 else [],
            "path": list(p),
            "slot_b": int(slot.get(n.flat_idx, 0)),
            "cum": cum,
            "cum_repr": f"{cum:.12g}",
            "logprob": float(n.logprob),
            "logprob_repr": f"{float(n.logprob):.12g}",
        }
    return out


def make_logits_and_bind(levels: dict, fanout: List[int]):
    """Placeholder logits (identity only) + bind dumped top-6 into _ACTIVE_LEVELS."""
    global _ACTIVE_LEVELS
    _ACTIVE_LEVELS = {}
    K = len(fanout)
    out: List[torch.Tensor] = []
    for k in range(K):
        t = torch.zeros(VOCAB, dtype=torch.float32)
        out.append(t)
        if k == 0:
            continue
        pairs = levels.get(str(k), [])
        if not pairs:
            continue
        lps, idxs = [], []
        for lp_repr, idx in pairs:
            lps.append(float(lp_repr) if isinstance(lp_repr, str) else float(lp_repr))
            idxs.append(int(idx))
        lp_t = torch.tensor(lps, dtype=torch.float64)
        idx_t = torch.tensor(idxs, dtype=torch.long)
        _ACTIVE_LEVELS[k] = (lp_t, idx_t, t)
    return out


def diff_round(rec: dict) -> dict:
    fanout = list(rec["fanout"])
    max_nodes = int(rec["max_nodes"])
    root = int(rec["known_next"])
    logits = make_logits_and_bind(rec["levels"], fanout)
    tree_mod.topk_masked = _patched_topk
    try:
        ns = build_tree_folded(logits, root, fanout, max_nodes, depth1_floor=True)
        nd = build_tree_folded_dynamic(logits, root, fanout, max_nodes, depth1_floor=True)
    finally:
        tree_mod.topk_masked = _ORIG_TOPK
    ss, sd = node_records(ns), node_records(nd)
    only_s = [ss[k] for k in sorted(ss.keys() - sd.keys())]
    only_d = [sd[k] for k in sorted(sd.keys() - ss.keys())]
    cum_diff = []
    for k in sorted(ss.keys() & sd.keys()):
        if ss[k]["cum"] != sd[k]["cum"]:
            cum_diff.append({
                "path": list(k),
                "static_cum": ss[k]["cum_repr"],
                "dynamic_cum": sd[k]["cum_repr"],
            })
    return {
        "prompt_id": rec["prompt_id"],
        "round": rec["round"],
        "known_next": root,
        "n_static": len(ns),
        "n_dynamic": len(nd),
        "widths_static": per_depth_widths(ns, len(fanout)),
        "widths_dynamic": per_depth_widths(nd, len(fanout)),
        "only_static": only_s,
        "only_dynamic": only_d,
        "cum_diff_same_path": cum_diff,
        "set_equal": (not only_s) and (not only_d) and (not cum_diff),
    }


def first_diff_for_prompt(rows: List[dict]) -> Optional[dict]:
    for rec in rows:
        d = diff_round(rec)
        if not d["set_equal"]:
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-jsonl", required=True)
    ap.add_argument("--prompt-ids", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-after-first-diffs", type=int, default=4)
    args = ap.parse_args()

    by_pid: Dict[str, List[dict]] = {}
    with open(args.dump_jsonl) as fh:
        for line in fh:
            rec = json.loads(line)
            if args.prompt_ids and str(rec["prompt_id"]) not in set(map(str, args.prompt_ids)):
                continue
            by_pid.setdefault(str(rec["prompt_id"]), []).append(rec)

    reports = []
    n_with_diff = 0
    for pid, rows in by_pid.items():
        rows = sorted(rows, key=lambda r: r["round"])
        first = first_diff_for_prompt(rows)
        reports.append({
            "prompt_id": pid,
            "n_rounds_dumped": len(rows),
            "first_diff": first,
            "all_equal": first is None,
        })
        if first is not None:
            n_with_diff += 1
            print(f"[diff] id={pid} round={first['round']} "
                  f"only_s={len(first['only_static'])} only_d={len(first['only_dynamic'])} "
                  f"cum_diff={len(first['cum_diff_same_path'])}")
            for side, key in (("only_static", "only_static"),
                              ("only_dynamic", "only_dynamic")):
                for n in first[key][:12]:
                    print(f"  {side}: tok={n['token']} depth={n['depth']} "
                          f"path={n['path']} slot_b={n['slot_b']} cum={n['cum_repr']}")
            if args.stop_after_first_diffs and n_with_diff >= args.stop_after_first_diffs:
                print(f"[diff] stop after {n_with_diff} prompts with set-diff")
                break
        else:
            print(f"[diff] id={pid} all rounds set-equal")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"reports": reports}, open(args.out, "w"), indent=2)
    print(f"[diff] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
