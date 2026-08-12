#!/usr/bin/env python3
"""3b(a): static gate#2 eval with per-round top-6 (lp, idx) dump — harness only.

Production decode/ is untouched. Dump is enough to rebuild both static and
dynamic trees offline (max fanout on gate#2 is 6).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from decode.common import (  # noqa: E402
    argmax_masked,
    cfg_attr,
    continuation_base,
    filter_prompts,
    load_base,
    load_head,
    make_image_inputs,
    topk_masked,
)
from decode.tree import (  # noqa: E402
    accept,
    build_mask_and_positions,
    build_tree_folded,
    reorg_kv_gather,
    tree_tokens,
)


def _lp_idx(logits_1d: torch.Tensor, k: int = 6):
    lp, idx = topk_masked(logits_1d.reshape(-1), k)
    # ≥10-digit precision via repr of python float from float64 cast
    pairs = []
    for i in range(k):
        pairs.append([repr(float(lp[i].double().item())), int(idx[i].item())])
    return pairs


@torch.no_grad()
def run_prompt_dump(base, head, processor, prompt, images_dir, fanout, max_nodes,
                    max_new, eos_id, dump_fh, device="cuda:0"):
    inputs = make_image_inputs(processor, prompt["question"],
                               images_dir / prompt["image"], device)
    out = base(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    P = past_kv.get_seq_length()
    h_anchor = out.hidden_states[-1][0, -1, :].clone()
    known_next = argmax_masked(out.logits[0, -1, :])
    del out
    torch.cuda.empty_cache()

    n_active = len(fanout)
    while n_active > 1 and fanout[n_active - 1] == 0:
        n_active -= 1
    fan_a = list(fanout[:n_active])
    emitted: List[int] = []
    rd = 0
    while len(emitted) < max_new:
        rd += 1
        cont_base = continuation_base(base, P)
        cond = base.get_input_embeddings()(
            torch.tensor([[known_next]], device=device, dtype=torch.long)
        )
        all_logits = head(
            h_anchor.view(1, 1, -1).half(), max_heads=n_active,
            skip_head0_lm_head=True, cond_embed=cond,
        )
        # dump top-6 per speculative head (heads 1..K-1); head0 unused in folded
        levels: Dict[str, list] = {}
        for k in range(1, n_active):
            if all_logits[k] is None:
                continue
            levels[str(k)] = _lp_idx(all_logits[k], 6)
        rec = {
            "prompt_id": str(prompt["id"]),
            "round": rd - 1,
            "known_next": int(known_next),
            "fanout": fan_a,
            "max_nodes": max_nodes,
            "levels": levels,  # head_k -> [[lp_repr, idx], ...] top-6
        }
        dump_fh.write(json.dumps(rec) + "\n")
        dump_fh.flush()

        nodes = build_tree_folded(all_logits, known_next, fan_a, max_nodes,
                                  depth1_floor=True)
        del all_logits, cond
        mask, pos = build_mask_and_positions(nodes, P, cont_base, base.dtype, device)
        toks = tree_tokens(nodes, device)
        v_out = base(
            input_ids=toks, attention_mask=mask, past_key_values=past_kv,
            position_ids=pos, use_cache=True, output_hidden_states=True,
        )
        accepted, _alen, bonus, _ = accept(nodes, v_out.logits[0], known_next)
        if not accepted:
            break
        accepted_tokens = [nodes[i].token for i in accepted]
        if eos_id in accepted_tokens:
            cut = accepted_tokens.index(eos_id) + 1
            emitted.extend(accepted_tokens[:cut])
            break
        emitted.extend(accepted_tokens)
        last = accepted[-1]
        h_anchor = v_out.hidden_states[-1][0, last, :].clone()
        reorg_kv_gather(v_out.past_key_values, P, accepted, device)
        past_kv = v_out.past_key_values
        P = past_kv.get_seq_length()
        known_next = bonus
        del v_out, nodes, mask, pos, toks
        torch.cuda.empty_cache()
        if bonus == eos_id:
            break
    del past_kv
    torch.cuda.empty_cache()
    return {"prompt_id": str(prompt["id"]), "emitted": len(emitted), "rounds": rd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="/scratch/li96/mz9869/eval_manifests/"
                    "manifest_old100_gate.json")
    ap.add_argument("--images-dir", default="/g/data/li96/mz9869/data/coco_subset")
    ap.add_argument("--fanout", type=int, nargs="+", default=[1, 6, 4, 2, 1])
    ap.add_argument("--max-nodes", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--only-indices", type=int, nargs="*", default=None,
                    help="If set, only these prompt indices (e.g. 10 58 78 85)")
    ap.add_argument("--dump-jsonl", required=True)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    prompts = filter_prompts(args.manifest, 80, 42, ordered=False)[: args.n_prompts]
    if args.only_indices is not None:
        prompts = [prompts[i] for i in args.only_indices]
        print(f"[3b-dump] subset indices={args.only_indices} n={len(prompts)}")
    else:
        print(f"[3b-dump] full n={len(prompts)} gate#2 fanout={args.fanout} "
              f"max_nodes={args.max_nodes}")

    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    head = load_head(
        args.ckpt,
        cfg_attr(base.config, "hidden_size"),
        cfg_attr(base.config, "vocab_size"),
    )
    eos_id = processor.tokenizer.eos_token_id
    images = Path(args.images_dir)
    dump_path = Path(args.dump_jsonl)
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    with open(dump_path, "w") as fh:
        for i, p in enumerate(prompts):
            print(f"[3b-dump] {i}/{len(prompts)} id={p['id']}")
            s = run_prompt_dump(
                base, head, processor, p, images,
                args.fanout, args.max_nodes, args.max_new_tokens, eos_id, fh,
            )
            summaries.append(s)
            print(f"  rounds={s['rounds']} emitted={s['emitted']}")

    summary_out = Path(args.summary_out) if args.summary_out else dump_path.with_suffix(
        ".summary.json")
    json.dump({"n": len(summaries), "fanout": args.fanout, "max_nodes": args.max_nodes,
               "prompts": summaries}, open(summary_out, "w"), indent=2)
    print(f"[3b-dump] wrote {dump_path} and {summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
