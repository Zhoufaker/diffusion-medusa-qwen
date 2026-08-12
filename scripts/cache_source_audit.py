"""cache_source_audit.py — quantitative cache-vs-eval domain audit.

Resolves the "what is the cache?" dispute with numbers instead of eyeballing a
handful of decodes.

Findings that shape the method (see code below):
  * cached_data/qwen25vl_long/manifest.json has NO per-sample source/dataset
    field (only n_cached/complete). Each <int>.pt holds only {hidden (L,3584),
    tokens (L,)}. So an authoritative group-by source is IMPOSSIBLE from the
    artifacts — we MUST detokenize `tokens` and classify by keywords.
  * eval = llava_subset_2k.json: 2000 {id, image (COCO), conversations[human,gpt]}.

What it reports (one run):
  1. chart-describing fraction for cache and eval, each with a 95% Wilson CI,
     plus the difference (with a normal-approx CI) = the domain-OOD magnitude.
  2. per-keyword firing counts (transparency / sensitivity of the classifier).
  3. source distribution — only if a per-sample source is ever found (it is not,
     in the current artifacts; reported as "unavailable" so nobody over-claims).
  4. answer-token-length distribution (cache vs eval; chart vs non-chart) +
     chart-fraction by eval length quartile -> tests the "the long subset is
     chart-heavy because chart answers live in the long tail" hypothesis.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

# chart-describing keyword classifier. Two tiers so we can report sensitivity:
#  STRONG  = phrases that essentially only occur in chart/statistic descriptions
#  WEAK    = single words that strongly co-occur with charts but can misfire
STRONG = [
    "statista", "this statistic", "the statistic", "statistic shows", "bar chart",
    "bar graph", "line graph", "pie chart", "line chart", "x-axis", "y-axis",
    "horizontal axis", "vertical axis", "the chart", "this chart", "the graph shows",
    "this graph", "in the graph", "data point", "the bar ", "number of ... in",
]
WEAK = ["chart", "graph", "axis", "plotted", "diagram", "percentage", "statistic"]


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def diff_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> Tuple[float, float, float]:
    p1, p2 = k1 / max(1, n1), k2 / max(1, n2)
    d = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / max(1, n1) + p2 * (1 - p2) / max(1, n2))
    return (d, d - z * se, d + z * se)


def classify(text: str) -> Tuple[bool, List[str]]:
    """Return (is_chart_strong, fired_keywords). Uses STRONG tier for the headline."""
    t = text.lower()
    fired = [kw for kw in STRONG if kw in t]
    return (len(fired) > 0, fired)


def classify_weak(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in WEAK)


def load_tokenizer(model_id: str):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    return proc.tokenizer


def cache_indices(manifest_path: Path, cache_dir: Path) -> int:
    m = json.load(open(manifest_path))
    return int(m.get("n_cached") or m.get("total") or 0)


def extract_cache_answer(tokens: torch.Tensor, tok) -> Tuple[str, int]:
    """Return (answer_text, answer_token_len) for a cached sequence.

    Qwen chat: ... <|im_start|>assistant\\n <answer> <|im_end|>. We take the span
    after the LAST <|im_start|> (skip the 'assistant\\n' header ~2 tokens) up to
    the next <|im_end|>, decoded with specials stripped. answer_token_len counts
    text tokens only (image-pad tokens are never in the answer span)."""
    ids = tokens.tolist()
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    last_start = max((i for i, t in enumerate(ids) if t == im_start), default=None)
    if last_start is None:
        txt = tok.decode(ids, skip_special_tokens=True)
        return txt, len(ids)
    seg = ids[last_start + 1:]
    end = next((i for i, t in enumerate(seg) if t == im_end), len(seg))
    seg = seg[:end]
    # drop the 'assistant' + '\n' role header tokens at the front
    header = tok("assistant\n", add_special_tokens=False).input_ids
    if seg[:len(header)] == header:
        seg = seg[len(header):]
    return tok.decode(seg, skip_special_tokens=True), len(seg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/scratch/li96/mz9869/cached_data/qwen25vl_long")
    ap.add_argument("--manifest", default="/scratch/li96/mz9869/cached_data/qwen25vl_long/manifest.json")
    ap.add_argument("--eval-json", default="/g/data/li96/mz9869/data/llava_subset_2k.json")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--n-cache", type=int, default=500)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default="/scratch/li96/mz9869/medusa_outputs/cache_source_audit.json")
    ap.add_argument("--out-fig", default="/scratch/li96/mz9869/medusa_outputs/cache_source_audit.png")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("[audit] loading tokenizer...", flush=True)
    tok = load_tokenizer(args.model_id)
    cache_dir = Path(args.cache_dir)

    # ---- CACHE sample ----
    n_total = cache_indices(Path(args.manifest), cache_dir)
    print(f"[audit] cache n_cached = {n_total}", flush=True)
    cand = list(range(n_total))
    rng.shuffle(cand)
    cache_recs = []
    for idx in cand:
        if len(cache_recs) >= args.n_cache:
            break
        p = cache_dir / f"{idx}.pt"
        if not p.exists():
            continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        ans, alen = extract_cache_answer(d["tokens"], tok)
        is_chart, fired = classify(ans)
        cache_recs.append({"idx": idx, "answer_len": alen, "is_chart": is_chart,
                           "weak": classify_weak(ans), "fired": fired})
    print(f"[audit] cache sampled = {len(cache_recs)}", flush=True)

    # ---- EVAL sample ----
    ev = json.load(open(args.eval_json))
    ev_idx = list(range(len(ev)))
    rng.shuffle(ev_idx)
    ev_idx = ev_idx[: args.n_eval]
    eval_recs = []
    for i in ev_idx:
        conv = ev[i]["conversations"]
        ans = next((c["value"] for c in reversed(conv) if c.get("from") == "gpt"), "")
        alen = len(tok(ans, add_special_tokens=False).input_ids)
        is_chart, fired = classify(ans)
        eval_recs.append({"id": ev[i]["id"], "answer_len": alen, "is_chart": is_chart,
                          "weak": classify_weak(ans), "fired": fired})
    print(f"[audit] eval sampled = {len(eval_recs)}", flush=True)

    # ---- fractions + CIs ----
    ck = sum(r["is_chart"] for r in cache_recs); cn = len(cache_recs)
    ek = sum(r["is_chart"] for r in eval_recs); en = len(eval_recs)
    cp, clo, chi = wilson(ck, cn)
    ep, elo, ehi = wilson(ek, en)
    dd, dlo, dhi = diff_ci(ck, cn, ek, en)
    # weak-tier sensitivity
    ckw = sum(r["weak"] for r in cache_recs); ekw = sum(r["weak"] for r in eval_recs)
    cpw, _, _ = wilson(ckw, cn); epw, _, _ = wilson(ekw, en)

    # keyword firing breakdown
    from collections import Counter
    cfire = Counter(kw for r in cache_recs for kw in r["fired"])
    efire = Counter(kw for r in eval_recs for kw in r["fired"])

    print("\n================ CHART-DESCRIBING FRACTION ================")
    print(f"  CACHE (qwen25vl_long): {cp*100:5.1f}%  [95% CI {clo*100:.1f}, {chi*100:.1f}]  ({ck}/{cn})")
    print(f"  EVAL  (llava_2k)     : {ep*100:5.1f}%  [95% CI {elo*100:.1f}, {ehi*100:.1f}]  ({ek}/{en})")
    print(f"  DIFF  (cache - eval) : {dd*100:5.1f} pp [95% CI {dlo*100:.1f}, {dhi*100:.1f}]  <- domain-OOD magnitude")
    print(f"  (weak-tier sanity: cache {cpw*100:.1f}% vs eval {epw*100:.1f}%)")
    print(f"  cache keyword fires: {dict(cfire.most_common(12))}")
    print(f"  eval  keyword fires: {dict(efire.most_common(12))}")

    print("\n================ SOURCE DISTRIBUTION ================")
    print("  per-sample source: UNAVAILABLE — manifest has only {n_cached,complete};")
    print("  each .pt has only {hidden, tokens}. Reported via classifier above, not group-by.")

    # ---- length distributions + long-filter test ----
    import statistics as st
    c_lens = [r["answer_len"] for r in cache_recs]
    e_lens = [r["answer_len"] for r in eval_recs]
    e_chart_lens = [r["answer_len"] for r in eval_recs if r["is_chart"]]
    e_non_lens = [r["answer_len"] for r in eval_recs if not r["is_chart"]]
    print("\n================ ANSWER-LENGTH (tokens) ================")
    print(f"  cache: mean={st.mean(c_lens):.0f} med={st.median(c_lens):.0f} "
          f"p10={_pct(c_lens,10):.0f} p90={_pct(c_lens,90):.0f}")
    print(f"  eval : mean={st.mean(e_lens):.0f} med={st.median(e_lens):.0f} "
          f"p10={_pct(e_lens,10):.0f} p90={_pct(e_lens,90):.0f}")
    if e_chart_lens and e_non_lens:
        print(f"  eval chart answers  : mean_len={st.mean(e_chart_lens):.0f} (n={len(e_chart_lens)})")
        print(f"  eval non-chart      : mean_len={st.mean(e_non_lens):.0f} (n={len(e_non_lens)})")
    # chart fraction by eval length quartile (long-filter hypothesis)
    qs = [_pct(e_lens, q) for q in (25, 50, 75)]
    bins = {"Q1(short)": [], "Q2": [], "Q3": [], "Q4(long)": []}
    for r in eval_recs:
        L = r["answer_len"]
        b = "Q1(short)" if L <= qs[0] else "Q2" if L <= qs[1] else "Q3" if L <= qs[2] else "Q4(long)"
        bins[b].append(r["is_chart"])
    print("  chart fraction by eval answer-length quartile (tests long->chart enrichment):")
    qfrac = {}
    for b, v in bins.items():
        f = sum(v) / max(1, len(v)); qfrac[b] = f
        print(f"      {b:9s}: {f*100:5.1f}%  (n={len(v)})")

    # ---- figure ----
    _make_fig(args.out_fig, c_lens, e_lens, e_chart_lens, e_non_lens, qfrac,
              cp, ep)

    out = {
        "cache": {"n": cn, "chart_k": ck, "chart_frac": cp, "ci95": [clo, chi],
                  "weak_frac": cpw, "keyword_fires": dict(cfire)},
        "eval": {"n": en, "chart_k": ek, "chart_frac": ep, "ci95": [elo, ehi],
                 "weak_frac": epw, "keyword_fires": dict(efire)},
        "diff_cache_minus_eval": {"point": dd, "ci95": [dlo, dhi]},
        "source_groupby": "UNAVAILABLE (no per-sample source in manifest or .pt)",
        "answer_len": {
            "cache_mean": st.mean(c_lens), "eval_mean": st.mean(e_lens),
            "eval_chart_mean": (st.mean(e_chart_lens) if e_chart_lens else None),
            "eval_nonchart_mean": (st.mean(e_non_lens) if e_non_lens else None),
            "eval_chart_frac_by_quartile": qfrac,
        },
        "classifier": {"strong": STRONG, "weak": WEAK},
        "seed": args.seed,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"\n[save] {args.out_json}\n[save] {args.out_fig}")
    return 0


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    f = int(k)
    return s[f] if f + 1 >= len(s) else s[f] + (k - f) * (s[f + 1] - s[f])


def _make_fig(path, c_lens, e_lens, e_chart, e_non, qfrac, cp, ep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    mx = int(max(max(c_lens, default=1), max(e_lens, default=1)))
    bins = list(range(0, mx + 40, 20))

    ax[0, 0].hist(c_lens, bins=bins, alpha=0.6, label="cache (qwen25vl_long)", color="C1")
    ax[0, 0].hist(e_lens, bins=bins, alpha=0.6, label="eval (llava_2k)", color="C0")
    ax[0, 0].set_title("answer-length distribution: cache vs eval")
    ax[0, 0].set_xlabel("answer tokens"); ax[0, 0].set_ylabel("count"); ax[0, 0].legend()

    if e_chart and e_non:
        ax[0, 1].hist(e_non, bins=bins, alpha=0.6, label="eval non-chart", color="C0")
        ax[0, 1].hist(e_chart, bins=bins, alpha=0.7, label="eval chart", color="C3")
        ax[0, 1].set_title("eval: chart answers live in the long tail?")
        ax[0, 1].set_xlabel("answer tokens"); ax[0, 1].set_ylabel("count"); ax[0, 1].legend()

    qs = list(qfrac.keys()); fs = [qfrac[k] * 100 for k in qs]
    ax[1, 0].bar(qs, fs, color="C2")
    ax[1, 0].set_title("eval chart-fraction by answer-length quartile")
    ax[1, 0].set_ylabel("chart %")
    for i, f in enumerate(fs):
        ax[1, 0].text(i, f, f"{f:.1f}%", ha="center", va="bottom")

    ax[1, 1].bar(["cache", "eval"], [cp * 100, ep * 100], color=["C1", "C0"])
    ax[1, 1].set_title("chart-describing fraction")
    ax[1, 1].set_ylabel("chart %")
    for i, f in enumerate([cp * 100, ep * 100]):
        ax[1, 1].text(i, f, f"{f:.1f}%", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
