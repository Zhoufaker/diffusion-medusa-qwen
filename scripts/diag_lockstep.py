"""diag_lockstep.py — round-by-round chain vs tree(width-1) divergence finder.

Runs the validated chain decoder (scripts.eval_acceptance semantics) and the
tree(width-1) decoder on ONE prompt, logging per-round internal state, then
reports the FIRST round whose (candidates, base_preds, accept_len, bonus) differ
and dumps the verify-logits top-2 margin there. Pinpoints whether the regression
divergence is a near-tie fp16 effect or a genuine logic difference.

Both decoders are reimplemented INLINE here (deterministic, same model) so we can
capture per-round records that the production functions don't expose.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/scratch/li96/mz9869/tmp_hf_download")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decode.common import (  # noqa: E402
    argmax_masked,
    continuation_base,
    filter_prompts,
    load_base,
    load_head,
    make_image_inputs,
    mask_phantom_,
    cfg_attr,
)

IMAGE = "000000055049.jpg"
MAX_NEW = 80


def _margin(logits_1d):
    t2 = mask_phantom_(logits_1d).float().topk(2).values
    return (t2[0] - t2[1]).item()


@torch.no_grad()
def run_chain(base, head, processor, prompt, images_dir, eos_id, device="cuda:0"):
    K = head.num_heads
    inputs = make_image_inputs(processor, prompt["question"], images_dir / prompt["image"], device)
    out = base(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    prefill_len = past_kv.get_seq_length()
    h_t = out.hidden_states[-1][0, -1, :].clone()
    base_pred_t = argmax_masked(out.logits[0, -1, :])
    del out
    emitted, recs = [], []
    while len(emitted) < MAX_NEW:
        P_round = past_kv.get_seq_length()   # capture BEFORE verify mutates the cache
        all_logits = head(h_t.view(1, 1, -1).half())
        candidates = [argmax_masked(L[0, 0]) for L in all_logits]
        v_in = torch.tensor([candidates], device=device, dtype=torch.long)
        v_out = base(input_ids=v_in, past_key_values=past_kv, use_cache=True)
        v_logits = v_out.logits[0]
        base_preds = [base_pred_t] + [argmax_masked(v_logits[i]) for i in range(K - 1)]
        accept_len = 0
        for i in range(K):
            if candidates[i] == base_preds[i]:
                accept_len = i + 1
            else:
                break
        bonus = base_pred_t if accept_len == 0 else argmax_masked(v_logits[accept_len - 1])
        margins = [_margin(v_logits[i]) for i in range(K)]
        recs.append((tuple(candidates), tuple(base_preds), accept_len, bonus, P_round, margins))
        emitted_round = candidates[:accept_len] + [bonus]
        prev = len(emitted)
        emitted.extend(emitted_round)
        if eos_id in emitted_round:
            break
        v_kv = v_out.past_key_values
        v_kv.crop(prefill_len + prev + accept_len)
        del v_out
        b_out = base(input_ids=torch.tensor([[bonus]], device=device), past_key_values=v_kv,
                     use_cache=True, output_hidden_states=True)
        past_kv = b_out.past_key_values
        h_t = b_out.hidden_states[-1][0, -1, :].clone()
        base_pred_t = argmax_masked(b_out.logits[0, -1, :])
        del b_out
    return emitted, recs


@torch.no_grad()
def run_tree_w1(base, head, processor, prompt, images_dir, eos_id, device="cuda:0"):
    from decode.tree import build_tree, build_mask_and_positions, tree_tokens, accept, reorg_kv_safe
    K = head.num_heads
    dtype = base.dtype
    inputs = make_image_inputs(processor, prompt["question"], images_dir / prompt["image"], device)
    out = base(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    P = past_kv.get_seq_length()
    h_t = out.hidden_states[-1][0, -1, :].clone()
    base_pred_root = argmax_masked(out.logits[0, -1, :])
    del out
    emitted, recs = [], []
    while len(emitted) < MAX_NEW:
        cont_base = continuation_base(base, P)
        all_logits = head(h_t.view(1, 1, -1).half())
        nodes = build_tree(all_logits, [1, 1, 1], 3, depth1_floor=False)
        mask, pos = build_mask_and_positions(nodes, P, cont_base, dtype, device)
        toks = tree_tokens(nodes, device)
        v_out = base(input_ids=toks, attention_mask=mask, past_key_values=past_kv,
                     position_ids=pos, use_cache=True)
        v_logits = v_out.logits[0]
        accepted, accept_len, bonus, _ = accept(nodes, v_logits, base_pred_root)
        # reconstruct chain-comparable records
        candidates = [n.token for n in sorted(nodes, key=lambda x: x.depth)]
        base_preds = [base_pred_root] + [argmax_masked(v_logits[i]) for i in range(K - 1)]
        margins = [_margin(v_logits[i]) for i in range(len(nodes))]
        recs.append((tuple(candidates), tuple(base_preds), accept_len, bonus, P, margins))
        accepted_tokens = [nodes[i].token for i in accepted]
        emitted_round = accepted_tokens + [bonus]
        emitted.extend(emitted_round)
        if eos_id in emitted_round:
            break
        past_kv = v_out.past_key_values
        reorg_kv_safe(base, past_kv, P, accepted_tokens, cont_base, device)
        del v_out
        P = past_kv.get_seq_length()
        cb = continuation_base(base, P)
        b_out = base(input_ids=torch.tensor([[bonus]], device=device), past_key_values=past_kv,
                     position_ids=torch.tensor([[cb]], device=device),
                     use_cache=True, output_hidden_states=True)
        past_kv = b_out.past_key_values
        h_t = b_out.hidden_states[-1][0, -1, :].clone()
        base_pred_root = argmax_masked(b_out.logits[0, -1, :])
        P = past_kv.get_seq_length()
        del b_out
    return emitted, recs


def main() -> int:
    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    head = load_head("/scratch/li96/mz9869/medusa_outputs/linked_medusa_v1_full/ckpt_best.pt",
                     cfg_attr(base.config, "hidden_size"), cfg_attr(base.config, "vocab_size"))
    eos_id = processor.tokenizer.eos_token_id
    images_dir = Path("/g/data/li96/mz9869/data/coco_subset")
    prompt = next(p for p in filter_prompts("/g/data/li96/mz9869/data/llava_subset_2k.json", 80, 42)
                  if p["image"] == IMAGE)

    em_c, rec_c = run_chain(base, head, processor, prompt, images_dir, eos_id)
    em_t, rec_t = run_tree_w1(base, head, processor, prompt, images_dir, eos_id)
    print(f"\nchain emit={len(em_c)}  tree emit={len(em_t)}  rounds chain={len(rec_c)} tree={len(rec_t)}")
    print(f"emitted byte-identical: {em_c == em_t}")
    div = next((j for j in range(min(len(em_c), len(em_t))) if em_c[j] != em_t[j]), None)
    print(f"first emitted divergence @ {div}")

    R = min(len(rec_c), len(rec_t))
    first = None
    for r in range(R):
        cc, bc, ac, boc, pc, mc = rec_c[r]
        ct, bt, at, bot, pt, mt = rec_t[r]
        same = (cc == ct and bc == bt and ac == at and boc == bot and pc == pt)
        if not same and first is None:
            first = r
            print(f"\n=== FIRST ROUND DIVERGENCE @ round {r} ===")
            print(f"  P:           chain={pc}  tree={pt}  {'OK' if pc==pt else 'DIFFER!'}")
            print(f"  candidates:  chain={cc}  tree={ct}  {'OK' if cc==ct else 'DIFFER!'}")
            print(f"  base_preds:  chain={bc}  tree={bt}  {'OK' if bc==bt else 'DIFFER!'}")
            print(f"  accept_len:  chain={ac}  tree={at}  {'OK' if ac==at else 'DIFFER!'}")
            print(f"  bonus:       chain={boc} tree={bot} {'OK' if boc==bot else 'DIFFER!'}")
            print(f"  verify top-2 margins chain={['%.3e'%x for x in mc]}")
            print(f"  verify top-2 margins tree ={['%.3e'%x for x in mt]}")
    if first is None:
        print("\n  per-round records IDENTICAL up to min rounds (no logic divergence).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
